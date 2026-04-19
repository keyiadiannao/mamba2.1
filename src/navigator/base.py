from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.tree_builder import TreeNode


@dataclass
class NavigatorState:
    path: list[str] = field(default_factory=list)
    text_bytes_seen: int = 0
    relevance_score: float = 0.0
    hidden_summary: list[float] | None = None
    backend_metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "NavigatorState":
        return NavigatorState(
            path=list(self.path),
            text_bytes_seen=self.text_bytes_seen,
            relevance_score=self.relevance_score,
            hidden_summary=list(self.hidden_summary) if self.hidden_summary is not None else None,
            backend_metadata=dict(self.backend_metadata),
        )


def build_path_recursive_prompt_text(
    question: str,
    ancestors: Sequence[TreeNode],
    current: TreeNode,
    *,
    max_chars_segment: int = 240,
    max_chars_question: int = 512,
) -> str:
    """P1 path-recursive protocol: single structured string for one navigator forward pass."""

    def _clip(text: str, max_chars: int) -> str:
        t = (text or "").strip().replace("\n", " ")
        if max_chars > 0 and len(t) > max_chars:
            return t[:max_chars] + "…"
        return t

    q = _clip(question, max_chars_question)
    segs = [_clip(a.text, max_chars_segment) for a in ancestors]
    cur = _clip(current.text, max_chars_segment)
    path_line = " -> ".join(segs) if segs else ""
    return f"[Q] {q}\n[PATH] {path_line}\n[NODE] {cur}"


def merge_path_summaries(previous_summary: list[float], current_summary: list[float]) -> list[float]:
    """Element-wise mean merge for path-sized summaries (shared by Mamba / sentence encoders)."""
    target_dim = min(len(previous_summary), len(current_summary))
    merged = [
        float((previous_summary[index] + current_summary[index]) / 2.0)
        for index in range(target_dim)
    ]
    if len(current_summary) > target_dim:
        merged.extend(float(value) for value in current_summary[target_dim:])
    return merged


class BaseNavigator(ABC):
    @abstractmethod
    def init_state(self) -> NavigatorState:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        question: str,
        node: TreeNode,
        state: NavigatorState,
        *,
        path_ancestor_nodes: Sequence[TreeNode] | None = None,
    ) -> NavigatorState:
        raise NotImplementedError


class MockMambaNavigator(BaseNavigator):
    """A lightweight stand-in for early pipeline integration."""

    def init_state(self) -> NavigatorState:
        return NavigatorState()

    def step(
        self,
        question: str,
        node: TreeNode,
        state: NavigatorState,
        *,
        path_ancestor_nodes: Sequence[TreeNode] | None = None,
    ) -> NavigatorState:
        next_state = state.clone()
        next_state.path.append(node.node_id)
        next_state.text_bytes_seen += len(node.text.encode("utf-8"))
        question_terms = {term.lower() for term in question.split() if term.strip()}
        text_terms = {term.lower() for term in node.text.split() if term.strip()}
        overlap = len(question_terms.intersection(text_terms))
        next_state.relevance_score = float(overlap)
        return next_state
