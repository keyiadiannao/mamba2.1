from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

from src.navigator import NavigatorState
from src.tree_builder import TreeNode


@dataclass
class ChildScore:
    node_id: str
    score: float


@dataclass
class RouteDecision:
    ordered_children: list[TreeNode]
    child_scores: list[ChildScore]


class BaseRouter(ABC):
    @abstractmethod
    def rank_children(
        self,
        question: str,
        parent: TreeNode,
        children: list[TreeNode],
        state: NavigatorState,
    ) -> RouteDecision:
        raise NotImplementedError


class RuleRouter(BaseRouter):
    """Sort children by lexical overlap with the question."""

    def rank_children(
        self,
        question: str,
        parent: TreeNode,
        children: list[TreeNode],
        state: NavigatorState,
    ) -> RouteDecision:
        question_terms = {term.lower() for term in question.split() if term.strip()}

        def score(node: TreeNode) -> tuple[float, str]:
            text_terms = {term.lower() for term in node.text.split() if term.strip()}
            return (float(len(question_terms.intersection(text_terms))), node.node_id)

        scored = [
            ChildScore(node_id=node.node_id, score=score(node)[0])
            for node in children
        ]
        score_map = {item.node_id: item.score for item in scored}
        ordered = sorted(children, key=lambda node: (score_map[node.node_id], node.node_id), reverse=True)
        scored = sorted(scored, key=lambda item: (item.score, item.node_id), reverse=True)
        return RouteDecision(ordered_children=ordered, child_scores=scored)


class CosineProbeRouter(BaseRouter):
    """A deterministic text-vector cosine router for Phase A comparisons."""

    def rank_children(
        self,
        question: str,
        parent: TreeNode,
        children: list[TreeNode],
        state: NavigatorState,
    ) -> RouteDecision:
        question_vector = self._text_vector(question)
        scored = [
            ChildScore(node_id=node.node_id, score=self._cosine_similarity(question_vector, self._text_vector(node.text)))
            for node in children
        ]
        score_map = {item.node_id: item.score for item in scored}
        ordered = sorted(children, key=lambda node: (score_map[node.node_id], node.node_id), reverse=True)
        scored = sorted(scored, key=lambda item: (item.score, item.node_id), reverse=True)
        return RouteDecision(ordered_children=ordered, child_scores=scored)

    def _text_vector(self, text: str) -> dict[str, float]:
        vector: dict[str, float] = {}
        for token in text.lower().split():
            cleaned = token.strip()
            if not cleaned:
                continue
            vector[cleaned] = vector.get(cleaned, 0.0) + 1.0
        return vector

    def _cosine_similarity(self, left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(numerator / (left_norm * right_norm))
