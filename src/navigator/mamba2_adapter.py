from __future__ import annotations

import importlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

from src.tree_builder import TreeNode

from .base import BaseNavigator, NavigatorState, build_path_recursive_prompt_text, merge_path_summaries


def _extract_last_hidden(outputs: Any) -> Any:
    """Pick (B, D) last-token vector from HF-style outputs, raw tensors, or tuples."""
    import torch

    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    elif isinstance(outputs, tuple) and outputs:
        hidden = None
        for item in outputs:
            if isinstance(item, torch.Tensor) and item.dim() >= 2:
                hidden = item
                break
        if hidden is None:
            raise TypeError(f"No tensor found in outputs tuple (len={len(outputs)}).")
    else:
        hidden = outputs
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"Expected hidden states as torch.Tensor, got {type(hidden)!r}.")
    if hidden.dim() != 3:
        raise ValueError(f"Expected 3D hidden states (B, S, D), got shape {tuple(hidden.shape)}.")
    return hidden[:, -1, :]


@dataclass
class Mamba2RuntimeConfig:
    backend: str = "mamba_ssm"
    model_name: str = "mamba2"
    load_strategy: str = "smoke_block"
    pretrained_checkpoint: str | None = None
    tokenizer_name: str | None = None
    device: str = "cuda"
    dtype: str = "float16"
    max_tokens_per_node: int = 512
    dependency_module: str = "mamba_ssm"
    d_model: int = 64
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    # Phase 1: keep each encode forward isolated from cross-call SSM caches (snapshot-safe).
    use_ssm_continuity: bool = False
    query_cache_max_size: int = 2048
    # P1 path-recursive protocol (see docs §5.0): one HF forward on [Q]/[PATH]/[NODE] text per step.
    path_recursive_prompt: bool = False
    path_prompt_max_chars_per_segment: int = 240
    path_prompt_max_question_chars: int = 512


class Mamba2Navigator(BaseNavigator):
    """Minimal real-backend integration for mamba2-style navigation smoke tests.

    ``NavigatorState.relevance_score`` is the navigator's own prior (lexical + semantic
    bonus here); router scores in ``RouteDecision`` are computed separately in the controller.
    """

    @property
    def uses_path_recursive_prompt(self) -> bool:
        return bool(self.config.path_recursive_prompt)

    def __init__(self, config: Mamba2RuntimeConfig | None = None) -> None:
        self.config = config or Mamba2RuntimeConfig()
        self._dependency = None
        self._torch = None
        self._device = None
        self._dtype = None
        self._summary_dim = self.config.d_model
        self._embedding = None
        self._model = None
        self._tokenizer = None
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()

    def clear_cache(self) -> None:
        """Drop cached question encodings (e.g. after a batch) to cap memory."""
        self._query_cache.clear()

    def _clear_ssm_ephemeral_state(self) -> None:
        """Best-effort clear of cross-call SSM caches when Phase 1 isolation is enabled."""
        if self.config.use_ssm_continuity or self._model is None:
            return
        reset_fn = getattr(self._model, "reset_state", None)
        if reset_fn is None:
            reset_fn = getattr(self._model, "reset_cache", None)
        if callable(reset_fn):
            reset_fn()

    def init_state(self) -> NavigatorState:
        self._ensure_runtime_ready()
        return NavigatorState(
            hidden_summary=[0.0] * self._summary_dim,
            backend_metadata={
                "backend": self.config.backend,
                "model_name": self.config.model_name,
                "load_strategy": self.config.load_strategy,
            },
        )

    def step(
        self,
        question: str,
        node: TreeNode,
        state: NavigatorState,
        *,
        path_ancestor_nodes: Sequence[TreeNode] | None = None,
    ) -> NavigatorState:
        torch = self._ensure_runtime_ready()
        next_state = state.clone()
        next_state.path.append(node.node_id)
        next_state.text_bytes_seen += len(node.text.encode("utf-8"))

        if self.config.path_recursive_prompt:
            if self.config.load_strategy != "hf_pretrained":
                raise RuntimeError(
                    "path_recursive_prompt requires navigator_load_strategy=hf_pretrained "
                    "(single structured forward per step)."
                )
            structured = build_path_recursive_prompt_text(
                question,
                tuple(path_ancestor_nodes or ()),
                node,
                max_chars_segment=int(self.config.path_prompt_max_chars_per_segment),
                max_chars_question=int(self.config.path_prompt_max_question_chars),
            )
            node_summary = self._encode_text_with_hf_model(structured, None, skip_merge=True)
        else:
            node_summary = self._encode_text(node.text, state.hidden_summary)
        query_summary = self._encode_query(question)
        lexical_overlap = self._lexical_overlap(question, node.text)
        semantic_bonus = max(self._cosine_similarity(query_summary, node_summary), 0.0)

        next_state.hidden_summary = node_summary
        next_state.relevance_score = lexical_overlap + semantic_bonus
        meta = {
            "backend": self.config.backend,
            "model_name": self.config.model_name,
            "load_strategy": self.config.load_strategy,
            "device": str(self._device),
            "dtype": str(self._dtype).replace("torch.", ""),
            "semantic_bonus": float(semantic_bonus),
            "lexical_overlap": float(lexical_overlap),
        }
        if self.config.path_recursive_prompt:
            meta["path_recursive_prompt"] = True
            meta["path_depth"] = len(path_ancestor_nodes or ())
        next_state.backend_metadata = meta
        return next_state

    def _ensure_runtime_ready(self) -> Any:
        dependency = self._ensure_dependency_available()
        if self._torch is None:
            self._torch = importlib.import_module("torch")

        if self._device is None:
            use_cuda = self.config.device.startswith("cuda") and self._torch.cuda.is_available()
            self._device = self._torch.device(self.config.device if use_cuda else "cpu")

        if self._dtype is None:
            self._dtype = self._resolve_dtype(self.config.dtype)

        if self._embedding is None:
            if self.config.load_strategy == "hf_pretrained":
                self._load_hf_pretrained_runtime()
                return self._torch
            self._embedding = self._torch.nn.Embedding(256, self.config.d_model).to(
                device=self._device,
                dtype=self._dtype,
            )

        if self._model is None:
            mamba_cls = getattr(dependency, "Mamba2", None)
            if mamba_cls is None:
                mamba_module = importlib.import_module("mamba_ssm.modules.mamba2")
                mamba_cls = getattr(mamba_module, "Mamba2")

            self._model = mamba_cls(
                d_model=self.config.d_model,
                d_state=self.config.d_state,
                d_conv=self.config.d_conv,
                expand=self.config.expand,
            ).to(device=self._device, dtype=self._dtype)
            self._model.eval()

        return self._torch

    def _load_hf_pretrained_runtime(self) -> None:
        transformers = importlib.import_module("transformers")
        checkpoint = self.config.pretrained_checkpoint or self.config.model_name
        tokenizer_name = self.config.tokenizer_name or checkpoint

        if self._tokenizer is None:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

        if self._model is None:
            model_kwargs: dict[str, Any] = {}
            if self._device.type != "cpu":
                model_kwargs["dtype"] = self._dtype
            try:
                self._model = transformers.AutoModel.from_pretrained(checkpoint, **model_kwargs)
            except Exception:
                self._model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
                if hasattr(self._model, "backbone"):
                    self._model = self._model.backbone

            self._model.to(self._device)
            self._model.eval()
            self._summary_dim = int(getattr(self._model.config, "hidden_size", self.config.d_model))

    def _encode_query(self, question: str) -> list[float]:
        cached = self._query_cache.get(question)
        if cached is not None:
            self._query_cache.move_to_end(question)
            return list(cached)
        encoded = self._encode_text(question, None)
        self._query_cache[question] = encoded
        max_size = max(0, int(self.config.query_cache_max_size))
        while len(self._query_cache) > max_size:
            self._query_cache.popitem(last=False)
        return list(encoded)

    def _encode_text(self, text: str, previous_summary: list[float] | None) -> list[float]:
        torch = self._ensure_runtime_ready()
        if self.config.load_strategy == "hf_pretrained":
            return self._encode_text_with_hf_model(text, previous_summary)

        token_values = list(text.encode("utf-8"))[: self.config.max_tokens_per_node]
        if not token_values:
            token_values = [0]

        token_tensor = torch.tensor(token_values, device=self._device, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            embedded = self._embedding(token_tensor)
            embedded = embedded.to(dtype=self._dtype)

            if previous_summary is not None:
                summary_tensor = torch.tensor(
                    previous_summary[: self.config.d_model],
                    device=self._device,
                    dtype=self._dtype,
                ).view(1, 1, self.config.d_model)
                embedded = torch.cat([summary_tensor, embedded], dim=1)

            self._clear_ssm_ephemeral_state()
            outputs = self._model(embedded)
            summary = _extract_last_hidden(outputs).detach().float().cpu().squeeze(0).tolist()

        return [float(value) for value in summary]

    def _encode_text_with_hf_model(
        self,
        text: str,
        previous_summary: list[float] | None,
        *,
        skip_merge: bool = False,
    ) -> list[float]:
        torch = self._ensure_runtime_ready()
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_tokens_per_node,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with torch.no_grad():
            self._clear_ssm_ephemeral_state()
            outputs = self._model(**encoded, return_dict=True)
            summary = _extract_last_hidden(outputs).detach().float().cpu().squeeze(0).tolist()

        if not skip_merge and previous_summary is not None:
            summary = self._merge_summaries(previous_summary, summary)
        return [float(value) for value in summary]

    def _merge_summaries(self, previous_summary: list[float], current_summary: list[float]) -> list[float]:
        return merge_path_summaries(previous_summary, current_summary)

    def _resolve_dtype(self, dtype_name: str) -> Any:
        torch = self._torch
        if self._device is not None and self._device.type == "cpu":
            return torch.float32
        lowered = dtype_name.lower()
        if lowered in {"float16", "fp16", "half"}:
            return torch.float16
        if lowered in {"bfloat16", "bf16"}:
            return torch.bfloat16
        return torch.float32

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0

        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(numerator / (left_norm * right_norm))

    def _lexical_overlap(self, question: str, text: str) -> float:
        question_terms = {term.lower() for term in question.split() if term.strip()}
        text_terms = {term.lower() for term in text.split() if term.strip()}
        return float(len(question_terms.intersection(text_terms)))

    def _ensure_dependency_available(self) -> Any:
        if self._dependency is not None:
            return self._dependency

        try:
            self._dependency = importlib.import_module(self.config.dependency_module)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Navigator backend '{self.config.backend}' requires the optional dependency "
                f"'{self.config.dependency_module}'. Install it in the active environment "
                "before running real Mamba2-backed navigation."
            ) from exc

        return self._dependency
