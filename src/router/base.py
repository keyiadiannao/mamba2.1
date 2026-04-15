from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
from pathlib import Path

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


def _tokenize_text(text: str) -> list[str]:
    return [token.strip().lower() for token in text.split() if token.strip()]


def _text_vector(text: str) -> dict[str, float]:
    vector: dict[str, float] = {}
    for token in _tokenize_text(text):
        vector[token] = vector.get(token, 0.0) + 1.0
    return vector


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def extract_router_features(question: str, node: TreeNode, state: NavigatorState) -> dict[str, float]:
    question_terms = set(_tokenize_text(question))
    text_terms = set(_tokenize_text(node.text))
    lexical_overlap = float(len(question_terms.intersection(text_terms)))
    cosine_probe = _cosine_similarity(_text_vector(question), _text_vector(node.text))
    text_length_tokens = float(len(_tokenize_text(node.text)))
    parent_relevance = float(state.relevance_score)
    child_is_leaf = 1.0 if node.is_leaf else 0.0
    return {
        "lexical_overlap": lexical_overlap,
        "cosine_probe": cosine_probe,
        "text_length_tokens": text_length_tokens,
        "parent_relevance": parent_relevance,
        "child_is_leaf": child_is_leaf,
    }


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
        def score(node: TreeNode) -> tuple[float, str]:
            return (extract_router_features(question, node, state)["lexical_overlap"], node.node_id)

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
        scored = [
            ChildScore(
                node_id=node.node_id,
                score=extract_router_features(question, node, state)["cosine_probe"],
            )
            for node in children
        ]
        score_map = {item.node_id: item.score for item in scored}
        ordered = sorted(children, key=lambda node: (score_map[node.node_id], node.node_id), reverse=True)
        scored = sorted(scored, key=lambda item: (item.score, item.node_id), reverse=True)
        return RouteDecision(ordered_children=ordered, child_scores=scored)


class LearnedClassifierRouter(BaseRouter):
    """A lightweight linear scoring head loaded from a JSON checkpoint."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        with path.open("r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        self.feature_names: list[str] = list(checkpoint["feature_names"])
        self.weights: list[float] = [float(value) for value in checkpoint["weights"]]
        self.bias: float = float(checkpoint.get("bias", 0.0))

    def rank_children(
        self,
        question: str,
        parent: TreeNode,
        children: list[TreeNode],
        state: NavigatorState,
    ) -> RouteDecision:
        scored = []
        for node in children:
            features = extract_router_features(question, node, state)
            score_value = self.bias
            for feature_name, weight in zip(self.feature_names, self.weights):
                score_value += weight * float(features.get(feature_name, 0.0))
            scored.append(ChildScore(node_id=node.node_id, score=float(score_value)))

        score_map = {item.node_id: item.score for item in scored}
        ordered = sorted(children, key=lambda node: (score_map[node.node_id], node.node_id), reverse=True)
        scored = sorted(scored, key=lambda item: (item.score, item.node_id), reverse=True)
        return RouteDecision(ordered_children=ordered, child_scores=scored)
