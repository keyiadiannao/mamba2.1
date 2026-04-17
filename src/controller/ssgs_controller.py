from __future__ import annotations

from dataclasses import dataclass

from src.navigator import BaseNavigator, NavigatorState
from src.router import BaseRouter, ChildScore, RouteDecision
from src.routing.entity_match import apply_entity_boost, extract_question_entities
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
    entity_boost_alpha: float = 0.0
    # 0 = disabled. When >0, cap how many accept_evidence entries may come from leaves under the same
    # direct child of the document root (root_branch), so navigation can spread across root subtrees.
    evidence_max_per_root_child: int = 0


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
        question_entities = extract_question_entities(question)
        trace = TraceRecord(
            routing_mode=self.config.routing_mode,
            context_source=self.config.context_source,
            navigator_type=self.navigator.__class__.__name__,
            visited_leaf_visits_ordered=[],
            visited_leaf_indices_deduped=[],
            entity_boost_alpha=float(self.config.entity_boost_alpha),
            question_entity_count=len(question_entities),
        )
        try:
            self._explore_node(
                question,
                tree.root,
                self.navigator.init_state(),
                trace,
                depth=0,
                question_entities=question_entities,
                root_branch_anchor=None,
            )
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
        question_entities: list[str],
        root_branch_anchor: str | None = None,
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
                cap = int(self.config.evidence_max_per_root_child or 0)
                if cap > 0 and root_branch_anchor is not None:
                    n_same = sum(
                        1
                        for ev in trace.event_log
                        if isinstance(ev, dict)
                        and ev.get("event") == "accept_evidence"
                        and ev.get("root_branch") == root_branch_anchor
                    )
                    if n_same >= cap:
                        trace.event_log.append(
                            {
                                "event": "reject_leaf_branch_cap",
                                "node_id": node.node_id,
                                "leaf_index": leaf_index,
                                "score": float(next_state.relevance_score),
                                "root_branch": root_branch_anchor,
                                "cap": cap,
                            }
                        )
                        return
                trace.evidence_texts.append(node.text)
                trace.evidence_node_ids.append(node.node_id)
                trace.event_log.append(
                    {
                        "event": "accept_evidence",
                        "node_id": node.node_id,
                        "leaf_index": leaf_index,
                        "score": float(next_state.relevance_score),
                        "root_branch": root_branch_anchor,
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
        trace.snapshot_push_count += 1
        trace.event_log.append(
            {
                "event": "snapshot_push",
                "node_id": node.node_id,
                "depth": depth,
                "stack_depth": len(snapshot.path),
            }
        )

        ordered = self.router.rank_children(question, node, node.children, next_state)
        ordered, entity_route_audit = self._apply_entity_boost(question_entities, node, ordered)
        trace.route_decisions.append(
            {
                "parent_node_id": node.node_id,
                "depth": depth,
                "ordered_child_ids": [child.node_id for child in ordered.ordered_children],
                "child_scores": [
                    {
                        "node_id": child_score.node_id,
                        "score": float(child_score.score),
                        "raw_router_score": entity_route_audit.get(child_score.node_id, {}).get(
                            "raw_router_score"
                        ),
                        "entity_match_score": entity_route_audit.get(child_score.node_id, {}).get(
                            "entity_match_score"
                        ),
                    }
                    for child_score in ordered.child_scores
                ],
            }
        )
        for child in ordered.ordered_children:
            before_evidence = len(trace.evidence_texts)
            child_anchor = child.node_id if depth == 0 else root_branch_anchor
            self._explore_node(
                question,
                child,
                snapshot.clone(),
                trace,
                depth + 1,
                question_entities=question_entities,
                root_branch_anchor=child_anchor,
            )
            if len(trace.evidence_texts) == before_evidence:
                trace.rollback_count += 1
                trace.snapshot_restore_count += 1
                trace.event_log.append(
                    {
                        "event": "rollback",
                        "from_child_node_id": child.node_id,
                        "to_parent_node_id": node.node_id,
                        "depth": depth,
                    }
                )
                trace.event_log.append(
                    {
                        "event": "snapshot_restore",
                        "from_child_node_id": child.node_id,
                        "to_parent_node_id": node.node_id,
                        "depth": depth,
                        "stack_depth": len(snapshot.path),
                    }
                )

            if len(trace.evidence_texts) >= self.config.max_evidence:
                break

    def _apply_entity_boost(
        self,
        question_entities: list[str],
        parent: TreeNode,
        route_decision: RouteDecision,
    ) -> tuple[RouteDecision, dict[str, dict[str, float | None]]]:
        child_lookup = {child.node_id: child for child in parent.children}
        scored_children = [
            {"node_id": child_score.node_id, "score": float(child_score.score)}
            for child_score in route_decision.child_scores
        ]
        boosted = apply_entity_boost(
            scored_children=scored_children,
            question_entities=question_entities,
            alpha=float(self.config.entity_boost_alpha),
            get_node_text=lambda node_id: child_lookup[node_id].text if node_id in child_lookup else "",
        )
        boosted_sorted = sorted(
            boosted,
            key=lambda item: (float(item.get("score", 0.0)), str(item.get("node_id", ""))),
            reverse=True,
        )

        audit_by_node: dict[str, dict[str, float | None]] = {}
        for item in boosted_sorted:
            nid = str(item.get("node_id", ""))
            audit_by_node[nid] = {
                "entity_match_score": float(item["entity_match_score"])
                if "entity_match_score" in item
                else None,
                "raw_router_score": float(item["raw_router_score"])
                if "raw_router_score" in item
                else None,
            }

        ordered_children = [
            child_lookup[item["node_id"]]
            for item in boosted_sorted
            if item.get("node_id") in child_lookup
        ]
        child_scores = [
            ChildScore(node_id=str(item.get("node_id")), score=float(item.get("score", 0.0)))
            for item in boosted_sorted
        ]
        return RouteDecision(ordered_children=ordered_children, child_scores=child_scores), audit_by_node
