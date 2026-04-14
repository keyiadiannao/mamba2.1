from __future__ import annotations

from dataclasses import dataclass

from src.navigator import BaseNavigator, NavigatorState
from src.router import BaseRouter
from src.tracing import TraceRecord
from src.tree_builder import DocumentTree, TreeNode


@dataclass
class ControllerConfig:
    routing_mode: str = "rule"
    context_source: str = "t1_visited_leaves_ordered"
    max_evidence: int = 3
    min_relevance_score: float = 1.0


class SSGSController:
    def __init__(
        self,
        navigator: BaseNavigator,
        router: BaseRouter,
        config: ControllerConfig,
    ) -> None:
        self.navigator = navigator
        self.router = router
        self.config = config

    def run(self, question: str, tree: DocumentTree) -> TraceRecord:
        trace = TraceRecord(
            routing_mode=self.config.routing_mode,
            context_source=self.config.context_source,
            visited_leaf_visits_ordered=[],
            visited_leaf_indices_deduped=[],
        )
        self._explore_node(question, tree.root, self.navigator.init_state(), trace)
        trace.nav_success = bool(trace.evidence_texts)
        trace.finalize()
        return trace

    def _explore_node(
        self,
        question: str,
        node: TreeNode,
        state: NavigatorState,
        trace: TraceRecord,
    ) -> None:
        if len(trace.evidence_texts) >= self.config.max_evidence:
            return

        next_state = self.navigator.step(question, node, state)
        trace.visited_node_ids.append(node.node_id)

        if node.is_leaf:
            visit_index = len(trace.visited_leaf_visits_ordered or [])
            cast_visits = trace.visited_leaf_visits_ordered or []
            cast_visits.append(visit_index)
            trace.visited_leaf_visits_ordered = cast_visits

            cast_deduped = trace.visited_leaf_indices_deduped or []
            if visit_index not in cast_deduped:
                cast_deduped.append(visit_index)
            trace.visited_leaf_indices_deduped = cast_deduped

            if next_state.relevance_score >= self.config.min_relevance_score:
                trace.evidence_texts.append(node.text)
            return

        snapshot = next_state.clone()
        trace.snapshot_stack_max_depth = max(trace.snapshot_stack_max_depth, len(snapshot.path))

        ordered = self.router.rank_children(question, node, node.children, next_state)
        for child in ordered.ordered_children:
            before_evidence = len(trace.evidence_texts)
            self._explore_node(question, child, snapshot.clone(), trace)
            if len(trace.evidence_texts) == before_evidence:
                trace.rollback_count += 1

            if len(trace.evidence_texts) >= self.config.max_evidence:
                break
