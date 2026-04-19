"""Sentence-embedding navigator for Mamba-vs-MiniLM-style diagnostics.

Uses the same ``merge_path_summaries`` fusion as :class:`Mamba2Navigator` (HF path).
Default checkpoint: ``sentence-transformers/all-MiniLM-L6-v2`` (384-d, English-friendly).

Optional dependency: ``pip install sentence-transformers``
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from src.tree_builder import TreeNode

from .base import BaseNavigator, NavigatorState, merge_path_summaries


def _lexical_overlap(question: str, text: str) -> float:
    question_terms = {term.lower() for term in question.split() if term.strip()}
    text_terms = {term.lower() for term in text.split() if term.strip()}
    return float(len(question_terms.intersection(text_terms)))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


@dataclass
class SentenceTransformerNavigatorConfig:
    """Runtime for :class:`SentenceTransformerNavigator`."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"
    dtype: str = "float32"
    max_chars_per_node: int = 16000
    query_cache_max_size: int = 2048


class SentenceTransformerNavigator(BaseNavigator):
    """Per-node sentence embedding + same path-vector fusion as Mamba (HF + merge)."""

    def __init__(self, config: SentenceTransformerNavigatorConfig | None = None) -> None:
        self.config = config or SentenceTransformerNavigatorConfig()
        self._model = None
        self._embedding_dim = 0
        self._device: Any = None
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()

    def clear_cache(self) -> None:
        self._query_cache.clear()

    def _ensure_runtime_ready(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "SentenceTransformerNavigator requires `sentence-transformers`. "
                "Install: pip install sentence-transformers"
            ) from exc

        device_str = str(self.config.device)
        if device_str.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    device_str = "cpu"
            except (OSError, ImportError):
                device_str = "cpu"
        self._device = device_str

        self._model = SentenceTransformer(self.config.model_name, device=device_str)
        self._model.eval()
        self._embedding_dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def init_state(self) -> NavigatorState:
        self._ensure_runtime_ready()
        return NavigatorState(
            hidden_summary=[0.0] * self._embedding_dim,
            backend_metadata={
                "backend": "sentence_transformer",
                "model_name": self.config.model_name,
                "embedding_dim": self._embedding_dim,
            },
        )

    def step(self, question: str, node: TreeNode, state: NavigatorState) -> NavigatorState:
        self._ensure_runtime_ready()
        next_state = state.clone()
        next_state.path.append(node.node_id)
        next_state.text_bytes_seen += len(node.text.encode("utf-8"))

        text = node.text or ""
        if self.config.max_chars_per_node > 0 and len(text) > self.config.max_chars_per_node:
            text = text[: self.config.max_chars_per_node]

        node_summary = self._encode_node_text(text, state.hidden_summary)
        query_summary = self._encode_query(question)
        lexical_overlap = _lexical_overlap(question, node.text)
        semantic_bonus = max(_cosine_similarity(query_summary, node_summary), 0.0)

        next_state.hidden_summary = node_summary
        next_state.relevance_score = lexical_overlap + semantic_bonus
        next_state.backend_metadata = {
            "backend": "sentence_transformer",
            "model_name": self.config.model_name,
            "embedding_dim": self._embedding_dim,
            "device": str(self._device),
            "semantic_bonus": float(semantic_bonus),
            "lexical_overlap": float(lexical_overlap),
        }
        return next_state

    def _encode_query(self, question: str) -> list[float]:
        cached = self._query_cache.get(question)
        if cached is not None:
            self._query_cache.move_to_end(question)
            return list(cached)
        encoded = self._encode_node_text(question, None)
        self._query_cache[question] = encoded
        max_size = max(0, int(self.config.query_cache_max_size))
        while len(self._query_cache) > max_size:
            self._query_cache.popitem(last=False)
        return list(encoded)

    def _encode_node_text(self, text: str, previous_summary: list[float] | None) -> list[float]:
        self._ensure_runtime_ready()

        t = (text or "").strip()
        if not t:
            current = [0.0] * self._embedding_dim
        else:
            emb = self._model.encode(
                t,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            current = self._embedding_to_float_list(emb)

        if previous_summary is not None:
            current = merge_path_summaries(previous_summary, current)
        return current

    def _embedding_to_float_list(self, emb: Any) -> list[float]:
        if emb is None:
            return [0.0] * self._embedding_dim
        if isinstance(emb, list):
            return [float(x) for x in emb]
        if isinstance(emb, tuple):
            return [float(x) for x in emb]
        try:
            import numpy as np

            if isinstance(emb, np.ndarray):
                return [float(x) for x in np.asarray(emb, dtype=np.float64).flatten().tolist()]
        except ModuleNotFoundError:
            pass
        import torch

        if isinstance(emb, torch.Tensor):
            return [float(x) for x in emb.flatten().float().detach().cpu().tolist()]
        return [float(x) for x in torch.tensor(emb).flatten().float().tolist()]
