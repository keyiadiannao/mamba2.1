from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from src.tree_builder import TreeNode

from .base import BaseNavigator, NavigatorState


@dataclass
class Mamba2RuntimeConfig:
    backend: str = "mamba_ssm"
    model_name: str = "mamba2"
    device: str = "cuda"
    dtype: str = "float16"
    max_tokens_per_node: int = 512
    dependency_module: str = "mamba_ssm"
    d_model: int = 64
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2


class Mamba2Navigator(BaseNavigator):
    """Minimal real-backend integration for mamba2-style navigation smoke tests."""

    def __init__(self, config: Mamba2RuntimeConfig | None = None) -> None:
        self.config = config or Mamba2RuntimeConfig()
        self._dependency = None
        self._torch = None
        self._device = None
        self._dtype = None
        self._embedding = None
        self._model = None
        self._query_cache: dict[str, list[float]] = {}

    def init_state(self) -> NavigatorState:
        self._ensure_runtime_ready()
        return NavigatorState(
            hidden_summary=[0.0] * self.config.d_model,
            backend_metadata={"backend": self.config.backend, "model_name": self.config.model_name},
        )

    def step(self, question: str, node: TreeNode, state: NavigatorState) -> NavigatorState:
        torch = self._ensure_runtime_ready()
        next_state = state.clone()
        next_state.path.append(node.node_id)
        next_state.text_bytes_seen += len(node.text.encode("utf-8"))

        node_summary = self._encode_text(node.text, state.hidden_summary)
        query_summary = self._encode_query(question)
        lexical_overlap = self._lexical_overlap(question, node.text)
        semantic_bonus = max(self._cosine_similarity(query_summary, node_summary), 0.0)

        next_state.hidden_summary = node_summary
        next_state.relevance_score = lexical_overlap + semantic_bonus
        next_state.backend_metadata = {
            "backend": self.config.backend,
            "model_name": self.config.model_name,
            "device": str(self._device),
            "dtype": str(self._dtype).replace("torch.", ""),
            "semantic_bonus": float(semantic_bonus),
            "lexical_overlap": float(lexical_overlap),
        }
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

    def _encode_query(self, question: str) -> list[float]:
        if question not in self._query_cache:
            self._query_cache[question] = self._encode_text(question, None)
        return list(self._query_cache[question])

    def _encode_text(self, text: str, previous_summary: list[float] | None) -> list[float]:
        torch = self._ensure_runtime_ready()
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

            outputs = self._model(embedded)
            summary = outputs[:, -1, :].detach().float().cpu().squeeze(0).tolist()

        return [float(value) for value in summary]

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
