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
    max_depth: int = 8
    max_nodes: int = 64


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
            navigator_type=self.navigator.__class__.__name__,
            visited_leaf_visits_ordered=[],
            visited_leaf_indices_deduped=[],
        )
        try:
            self._explore_node(question, tree.root, self.navigator.init_state(), trace, depth=0)
        except Exception as exc:
            trace.context_build_error = str(exc)
            trace.failure_attribution = "nav_failure_or_mixed"
        trace.nav_success = bool(trace.evidence_texts)
        if trace.failure_attribution is None:
            trace.failure_attribution = None if trace.nav_success else "nav_failure_or_mixed"
        trace.finalize()
        return trace

    def _explore_node(
        self,
        question: str,
        node: TreeNode,
        state: NavigatorState,
        trace: TraceRecord,
        depth: int,
    ) -> None:
        if len(trace.evidence_texts) >= self.config.max_evidence:
            return
        if len(trace.visited_node_ids) >= self.config.max_nodes:
            trace.event_log.append({"event": "max_nodes_reached", "node_id": node.node_id, "depth": depth})
            return
        if depth > self.config.max_depth:
            trace.event_log.append({"event": "max_depth_reached", "node_id": node.node_id, "depth": depth})
            return

        next_state = self.navigator.step(question, node, state)
        trace.visited_node_ids.append(node.node_id)
        trace.node_scores[node.node_id] = float(next_state.relevance_score)
        trace.event_log.append(
            {
                "event": "visit_node",
                "node_id": node.node_id,
                "depth": depth,
                "score": float(next_state.relevance_score),
            }
        )

        if node.is_leaf:
            leaf_index = node.metadata.get("leaf_index")
            if not isinstance(leaf_index, int):
                leaf_index = len(trace.visited_leaf_visits_ordered or [])

            cast_visits = trace.visited_leaf_visits_ordered or []
            cast_visits.append(leaf_index)
            trace.visited_leaf_visits_ordered = cast_visits

            cast_deduped = trace.visited_leaf_indices_deduped or []
            if leaf_index not in cast_deduped:
                cast_deduped.append(leaf_index)
            trace.visited_leaf_indices_deduped = cast_deduped

            if next_state.relevance_score >= self.config.min_relevance_score:
                trace.evidence_texts.append(node.text)
                trace.evidence_node_ids.append(node.node_id)
                trace.event_log.append(
                    {
                        "event": "accept_evidence",
                        "node_id": node.node_id,
                        "leaf_index": leaf_index,
                        "score": float(next_state.relevance_score),
                    }
                )
            else:
                trace.event_log.append(
                    {
                        "event": "reject_leaf",
                        "node_id": node.node_id,
                        "leaf_index": leaf_index,
                        "score": float(next_state.relevance_score),
                    }
                )
            return

        snapshot = next_state.clone()
        trace.snapshot_stack_max_depth = max(trace.snapshot_stack_max_depth, len(snapshot.path))

        ordered = self.router.rank_children(question, node, node.children, next_state)
        trace.route_decisions.append(
            {
                "parent_node_id": node.node_id,
                "depth": depth,
                "ordered_child_ids": [child.node_id for child in ordered.ordered_children],
                "child_scores": [
                    {"node_id": child_score.node_id, "score": float(child_score.score)}
                    for child_score in ordered.child_scores
                ],
            }
        )
        for child in ordered.ordered_children:
            before_evidence = len(trace.evidence_texts)
            self._explore_node(question, child, snapshot.clone(), trace, depth + 1)
            if len(trace.evidence_texts) == before_evidence:
                trace.rollback_count += 1
                trace.event_log.append(
                    {
                        "event": "rollback",
                        "from_child_node_id": child.node_id,
                        "to_parent_node_id": node.node_id,
                        "depth": depth,
                    }
                )

            if len(trace.evidence_texts) >= self.config.max_evidence:
                break
