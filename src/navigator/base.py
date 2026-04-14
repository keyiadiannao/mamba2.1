from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.tree_builder import TreeNode


@dataclass
class NavigatorState:
    path: list[str] = field(default_factory=list)
    text_bytes_seen: int = 0
    relevance_score: float = 0.0

    def clone(self) -> "NavigatorState":
        return NavigatorState(
            path=list(self.path),
            text_bytes_seen=self.text_bytes_seen,
            relevance_score=self.relevance_score,
        )


class BaseNavigator(ABC):
    @abstractmethod
    def init_state(self) -> NavigatorState:
        raise NotImplementedError

    @abstractmethod
    def step(self, question: str, node: TreeNode, state: NavigatorState) -> NavigatorState:
        raise NotImplementedError


class MockMambaNavigator(BaseNavigator):
    """A lightweight stand-in for early pipeline integration."""

    def init_state(self) -> NavigatorState:
        return NavigatorState()

    def step(self, question: str, node: TreeNode, state: NavigatorState) -> NavigatorState:
        next_state = state.clone()
        next_state.path.append(node.node_id)
        next_state.text_bytes_seen += len(node.text.encode("utf-8"))
        question_terms = {term.lower() for term in question.split() if term.strip()}
        text_terms = {term.lower() for term in node.text.split() if term.strip()}
        overlap = len(question_terms.intersection(text_terms))
        next_state.relevance_score = float(overlap)
        return next_state
