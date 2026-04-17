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


# Keys produced by ``extract_router_features`` (linear learned head must reference these only).
ROUTER_LINEAR_FEATURE_KEYS = frozenset(
    {
        "lexical_overlap",
        "cosine_probe",
        "text_length_tokens",
        "parent_relevance",
        "child_is_leaf",
    }
)


def extract_router_features(
    question: str,
    node: TreeNode,
    state: NavigatorState,
    *,
    question_terms: set[str] | None = None,
    question_vector: dict[str, float] | None = None,
) -> dict[str, float]:
    """Per-child routing features. Pass ``question_terms`` / ``question_vector`` to avoid
    re-tokenizing / re-vectorizing the question for every sibling (same stable outputs).
    """
    q_terms = set(_tokenize_text(question)) if question_terms is None else set(question_terms)
    q_vec = _text_vector(question) if question_vector is None else question_vector
    text_terms = set(_tokenize_text(node.text))
    lexical_overlap = float(len(q_terms.intersection(text_terms)))
    cosine_probe = _cosine_similarity(q_vec, _text_vector(node.text))
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


def _build_route_decision(children: list[TreeNode], scored: list[ChildScore]) -> RouteDecision:
    """Stable ordering shared by all routers: score desc, ``node_id`` desc on ties."""
    scored_sorted = sorted(scored, key=lambda item: (item.score, item.node_id), reverse=True)
    score_map = {item.node_id: item.score for item in scored_sorted}
    ordered = sorted(children, key=lambda node: (score_map[node.node_id], node.node_id), reverse=True)
    return RouteDecision(ordered_children=ordered, child_scores=scored_sorted)


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
    """Rank children primarily by lexical overlap with the question.

    Optional ``cosine_weight`` blends in bag-of-words cosine (same feature as CosineProbeRouter)
    to break ties and nudge order when many siblings share the same small integer overlap —
    a common case on real corpus trees where pure lexical routing under-explores.
    Defaults preserve historical behavior (lexical-only).
    """

    def __init__(
        self,
        *,
        lexical_weight: float = 1.0,
        cosine_weight: float = 0.0,
    ) -> None:
        self.lexical_weight = float(lexical_weight)
        self.cosine_weight = float(cosine_weight)

    def rank_children(
        self,
        question: str,
        parent: TreeNode,
        children: list[TreeNode],
        state: NavigatorState,
    ) -> RouteDecision:
        q_terms = set(_tokenize_text(question))
        q_vec = _text_vector(question)
        scored = []
        for node in children:
            feats = extract_router_features(
                question, node, state, question_terms=q_terms, question_vector=q_vec
            )
            score = self.lexical_weight * float(feats["lexical_overlap"]) + self.cosine_weight * float(
                feats["cosine_probe"]
            )
            scored.append(ChildScore(node_id=node.node_id, score=float(score)))
        return _build_route_decision(children, scored)


class CosineProbeRouter(BaseRouter):
    """A deterministic text-vector cosine router for Phase A comparisons."""

    def rank_children(
        self,
        question: str,
        parent: TreeNode,
        children: list[TreeNode],
        state: NavigatorState,
    ) -> RouteDecision:
        q_terms = set(_tokenize_text(question))
        q_vec = _text_vector(question)
        scored = [
            ChildScore(
                node_id=node.node_id,
                score=extract_router_features(
                    question, node, state, question_terms=q_terms, question_vector=q_vec
                )["cosine_probe"],
            )
            for node in children
        ]
        return _build_route_decision(children, scored)


class LearnedClassifierRouter(BaseRouter):
    """A lightweight linear scoring head loaded from a JSON checkpoint."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        with path.open("r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        self.feature_names: list[str] = list(checkpoint["feature_names"])
        self.weights: list[float] = [float(value) for value in checkpoint["weights"]]
        self.bias: float = float(checkpoint.get("bias", 0.0))
        if not self.feature_names:
            raise ValueError("Checkpoint feature_names must be non-empty.")
        if len(self.feature_names) != len(self.weights):
            raise ValueError(
                f"Checkpoint feature_names length {len(self.feature_names)} "
                f"does not match weights length {len(self.weights)}."
            )
        unknown = set(self.feature_names) - ROUTER_LINEAR_FEATURE_KEYS
        if unknown:
            raise ValueError(
                "Checkpoint contains unknown feature names (not produced by extract_router_features): "
                f"{sorted(unknown)}"
            )

    def score_from_features(self, features: dict[str, float]) -> float:
        """Linear logit used for ranking (same as ``rank_children`` per child)."""
        score_value = self.bias
        for feature_name, weight in zip(self.feature_names, self.weights):
            score_value += weight * float(features.get(feature_name, 0.0))
        return float(score_value)

    def rank_children(
        self,
        question: str,
        parent: TreeNode,
        children: list[TreeNode],
        state: NavigatorState,
    ) -> RouteDecision:
        q_terms = set(_tokenize_text(question))
        q_vec = _text_vector(question)
        scored = []
        for node in children:
            features = extract_router_features(
                question, node, state, question_terms=q_terms, question_vector=q_vec
            )
            scored.append(
                ChildScore(node_id=node.node_id, score=float(self.score_from_features(features)))
            )

        return _build_route_decision(children, scored)


class LearnedRootHybridRouter(BaseRouter):
    """Use learned linear head only at root, fallback to rule elsewhere.

    At root, optional **blend** with the same rule score used below root:
    ``score = (1 - blend_alpha) * s_rule + blend_alpha * s_learned``.
    ``blend_alpha=0`` is pure rule at root; ``1`` is pure learned (legacy behavior).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        fallback_lexical_weight: float = 1.0,
        fallback_cosine_weight: float = 0.0,
        blend_alpha: float = 0.25,
    ) -> None:
        self.root_router = LearnedClassifierRouter(checkpoint_path)
        self.fallback_router = RuleRouter(
            lexical_weight=fallback_lexical_weight,
            cosine_weight=fallback_cosine_weight,
        )
        self.blend_alpha = float(min(max(blend_alpha, 0.0), 1.0))

    def rank_children(
        self,
        question: str,
        parent: TreeNode,
        children: list[TreeNode],
        state: NavigatorState,
    ) -> RouteDecision:
        # ``state.path`` includes ``parent`` after navigator step in controller.
        if len(state.path) > 1:
            return self.fallback_router.rank_children(question, parent, children, state)

        if self.blend_alpha <= 0.0:
            return self.fallback_router.rank_children(question, parent, children, state)
        if self.blend_alpha >= 1.0:
            return self.root_router.rank_children(question, parent, children, state)

        q_terms = set(_tokenize_text(question))
        q_vec = _text_vector(question)
        a = self.blend_alpha
        scored: list[ChildScore] = []
        for node in children:
            feats = extract_router_features(
                question, node, state, question_terms=q_terms, question_vector=q_vec
            )
            s_rule = self.fallback_router.lexical_weight * float(feats["lexical_overlap"]) + (
                self.fallback_router.cosine_weight * float(feats["cosine_probe"])
            )
            s_learned = self.root_router.score_from_features(feats)
            score = (1.0 - a) * s_rule + a * s_learned
            scored.append(ChildScore(node_id=node.node_id, score=float(score)))
        return _build_route_decision(children, scored)
