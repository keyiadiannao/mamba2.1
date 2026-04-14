from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.navigator import NavigatorState
from src.tree_builder import TreeNode


@dataclass
class RouteDecision:
    ordered_children: list[TreeNode]


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

        def score(node: TreeNode) -> tuple[int, str]:
            text_terms = {term.lower() for term in node.text.split() if term.strip()}
            return (len(question_terms.intersection(text_terms)), node.node_id)

        ordered = sorted(children, key=score, reverse=True)
        return RouteDecision(ordered_children=ordered)
