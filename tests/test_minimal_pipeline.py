from __future__ import annotations

import unittest

from src.controller import ControllerConfig, SSGSController
from src.generator_bridge import build_generator_prompt
from src.navigator import MockMambaNavigator
from src.router import BaseRouter, ChildScore, RouteDecision, RuleRouter
from src.tracing import FROZEN_TRACE_FIELDS
from src.tree_builder import DocumentTree, TreeNode


def build_test_tree() -> DocumentTree:
    return DocumentTree(
        root=TreeNode(
            node_id="root",
            text="physics knowledge index",
            children=[
                TreeNode(
                    node_id="branch_relativity",
                    text="Einstein relativity branch",
                    children=[
                        TreeNode(
                            node_id="leaf_relativity_1",
                            text="Einstein proposed general relativity.",
                        )
                    ],
                )
            ],
        )
    )


class MinimalPipelineTest(unittest.TestCase):
    def test_trace_schema_contains_frozen_fields(self) -> None:
        for field_name in [
            "routing_mode",
            "context_source",
            "batch_id",
            "rollback_count",
            "snapshot_stack_max_depth",
            "snapshot_push_count",
            "snapshot_restore_count",
            "nav_wall_time_ms",
            "answer_f1",
        ]:
            self.assertIn(field_name, FROZEN_TRACE_FIELDS)

    def test_controller_exports_evidence_and_prompt(self) -> None:
        controller = SSGSController(
            navigator=MockMambaNavigator(),
            router=RuleRouter(),
            config=ControllerConfig(),
        )
        trace = controller.run("What did Einstein propose?", build_test_tree())
        prompt = build_generator_prompt("What did Einstein propose?", trace.evidence_texts)

        self.assertTrue(trace.nav_success)
        self.assertGreaterEqual(len(trace.evidence_texts), 1)
        self.assertIn("Einstein", prompt)
        self.assertIn("Return only the shortest final answer phrase.", prompt)
        self.assertGreaterEqual(len(trace.event_log), 1)
        self.assertGreaterEqual(len(trace.route_decisions), 1)
        self.assertIn("leaf_relativity_1", trace.evidence_node_ids)
        self.assertGreaterEqual(trace.snapshot_push_count, 1)
        self.assertGreaterEqual(trace.snapshot_restore_count, 0)
        self.assertTrue(any(event["event"] == "snapshot_push" for event in trace.event_log))

    def test_controller_entity_boost_reranks_children(self) -> None:
        class FixedOrderRouter(BaseRouter):
            def rank_children(self, question, parent, children, state):  # noqa: ANN001
                ordered_children = list(children)
                child_scores = [
                    ChildScore(node_id=children[0].node_id, score=0.0),
                    ChildScore(node_id=children[1].node_id, score=0.0),
                ]
                return RouteDecision(ordered_children=ordered_children, child_scores=child_scores)

        tree = DocumentTree(
            root=TreeNode(
                node_id="root",
                text="index",
                children=[
                    TreeNode(node_id="leaf_irrelevant", text="Newton discussed gravity."),
                    TreeNode(node_id="leaf_target", text="Einstein developed relativity."),
                ],
            )
        )
        controller = SSGSController(
            navigator=MockMambaNavigator(),
            router=FixedOrderRouter(),
            config=ControllerConfig(max_evidence=1, min_relevance_score=-1.0, entity_boost_alpha=1.0),
        )

        trace = controller.run("What did Einstein develop?", tree)
        self.assertEqual(trace.evidence_node_ids, ["leaf_target"])
        self.assertEqual(trace.question_entity_count, 1)
        self.assertEqual(trace.entity_boost_alpha, 1.0)
        route = trace.route_decisions[0]
        self.assertEqual(route["ordered_child_ids"][0], "leaf_target")
        score_by_node = {row["node_id"]: row for row in route["child_scores"]}
        self.assertEqual(score_by_node["leaf_target"]["entity_match_score"], 1.0)
        self.assertEqual(score_by_node["leaf_irrelevant"]["entity_match_score"], 0.0)

    def test_evidence_max_per_root_child_spreads_across_root_branches(self) -> None:
        class FixedOrderRouter(BaseRouter):
            def rank_children(self, question, parent, children, state):  # noqa: ANN001
                by_id = {c.node_id: c for c in children}
                if parent.node_id == "root":
                    order = ["branch_high", "branch_low", "branch_mid"]
                    score_by_id = {"branch_high": 2.0, "branch_low": 1.0, "branch_mid": 0.0}
                elif parent.node_id == "branch_high":
                    order = ["leaf_h1", "leaf_h2", "leaf_h3"]
                    score_by_id = {"leaf_h1": 3.0, "leaf_h2": 2.0, "leaf_h3": 1.0}
                elif parent.node_id == "branch_low":
                    order = ["leaf_l1", "leaf_l2"]
                    score_by_id = {"leaf_l1": 2.0, "leaf_l2": 1.0}
                elif parent.node_id == "branch_mid":
                    order = ["leaf_m1"]
                    score_by_id = {"leaf_m1": 1.0}
                else:
                    order = [c.node_id for c in children]
                    score_by_id = {}
                ordered = [by_id[nid] for nid in order if nid in by_id]
                for child in children:
                    if child not in ordered:
                        ordered.append(child)
                child_scores = [
                    ChildScore(node_id=c.node_id, score=float(score_by_id.get(c.node_id, 0.0)))
                    for c in ordered
                ]
                return RouteDecision(ordered_children=ordered, child_scores=child_scores)

        tree = DocumentTree(
            root=TreeNode(
                node_id="root",
                text="r",
                children=[
                    TreeNode(
                        node_id="branch_high",
                        text="h",
                        children=[
                            TreeNode(
                                node_id="leaf_h1",
                                text="Einstein relativity physics one",
                                metadata={"leaf_index": 1},
                            ),
                            TreeNode(
                                node_id="leaf_h2",
                                text="Einstein relativity physics two",
                                metadata={"leaf_index": 2},
                            ),
                            TreeNode(
                                node_id="leaf_h3",
                                text="Einstein relativity physics three",
                                metadata={"leaf_index": 3},
                            ),
                        ],
                    ),
                    TreeNode(
                        node_id="branch_low",
                        text="l",
                        children=[
                            TreeNode(
                                node_id="leaf_l1",
                                text="Einstein relativity physics four",
                                metadata={"leaf_index": 4},
                            ),
                            TreeNode(
                                node_id="leaf_l2",
                                text="Einstein relativity physics five",
                                metadata={"leaf_index": 5},
                            ),
                        ],
                    ),
                    TreeNode(
                        node_id="branch_mid",
                        text="m",
                        children=[
                            TreeNode(
                                node_id="leaf_m1",
                                text="Einstein relativity physics six",
                                metadata={"leaf_index": 6},
                            ),
                        ],
                    ),
                ],
            )
        )
        question = "Einstein relativity physics"

        trace_no_cap = SSGSController(
            navigator=MockMambaNavigator(),
            router=FixedOrderRouter(),
            config=ControllerConfig(
                max_evidence=3,
                min_relevance_score=1.0,
                evidence_max_per_root_child=0,
            ),
        ).run(question, tree)
        self.assertEqual(trace_no_cap.evidence_node_ids, ["leaf_h1", "leaf_h2", "leaf_h3"])

        trace_cap = SSGSController(
            navigator=MockMambaNavigator(),
            router=FixedOrderRouter(),
            config=ControllerConfig(
                max_evidence=3,
                min_relevance_score=1.0,
                evidence_max_per_root_child=1,
            ),
        ).run(question, tree)
        self.assertEqual(trace_cap.evidence_node_ids, ["leaf_h1", "leaf_l1", "leaf_m1"])
        self.assertGreaterEqual(
            sum(1 for ev in trace_cap.event_log if ev.get("event") == "reject_leaf_branch_cap"),
            1,
        )
        accepts = [ev for ev in trace_cap.event_log if ev.get("event") == "accept_evidence"]
        self.assertEqual(
            {ev.get("root_branch") for ev in accepts},
            {"branch_high", "branch_low", "branch_mid"},
        )

    def test_explore_top_m_root_children_allocates_budget_across_selected_roots(self) -> None:
        class FixedOrderRouter(BaseRouter):
            def rank_children(self, question, parent, children, state):  # noqa: ANN001
                by_id = {c.node_id: c for c in children}
                if parent.node_id == "root":
                    order = ["branch_high", "branch_low", "branch_mid"]
                    score_by_id = {"branch_high": 2.0, "branch_low": 1.0, "branch_mid": 0.0}
                elif parent.node_id == "branch_high":
                    order = ["leaf_h1", "leaf_h2", "leaf_h3"]
                    score_by_id = {"leaf_h1": 3.0, "leaf_h2": 2.0, "leaf_h3": 1.0}
                elif parent.node_id == "branch_low":
                    order = ["leaf_l1", "leaf_l2"]
                    score_by_id = {"leaf_l1": 2.0, "leaf_l2": 1.0}
                else:
                    order = [c.node_id for c in children]
                    score_by_id = {}
                ordered = [by_id[nid] for nid in order if nid in by_id]
                child_scores = [
                    ChildScore(node_id=c.node_id, score=float(score_by_id.get(c.node_id, 0.0)))
                    for c in ordered
                ]
                return RouteDecision(ordered_children=ordered, child_scores=child_scores)

        tree = DocumentTree(
            root=TreeNode(
                node_id="root",
                text="r",
                children=[
                    TreeNode(
                        node_id="branch_high",
                        text="h",
                        children=[
                            TreeNode(node_id="leaf_h1", text="Einstein relativity 1", metadata={"leaf_index": 1}),
                            TreeNode(node_id="leaf_h2", text="Einstein relativity 2", metadata={"leaf_index": 2}),
                            TreeNode(node_id="leaf_h3", text="Einstein relativity 3", metadata={"leaf_index": 3}),
                        ],
                    ),
                    TreeNode(
                        node_id="branch_low",
                        text="l",
                        children=[
                            TreeNode(node_id="leaf_l1", text="Einstein relativity 4", metadata={"leaf_index": 4}),
                            TreeNode(node_id="leaf_l2", text="Einstein relativity 5", metadata={"leaf_index": 5}),
                        ],
                    ),
                    TreeNode(
                        node_id="branch_mid",
                        text="m",
                        children=[
                            TreeNode(node_id="leaf_m1", text="Einstein relativity 6", metadata={"leaf_index": 6}),
                        ],
                    ),
                ],
            )
        )
        question = "Einstein relativity"

        trace_top_m = SSGSController(
            navigator=MockMambaNavigator(),
            router=FixedOrderRouter(),
            config=ControllerConfig(
                max_evidence=4,
                min_relevance_score=1.0,
                explore_top_m_root_children=2,
            ),
        ).run(question, tree)
        self.assertEqual(trace_top_m.evidence_node_ids, ["leaf_h1", "leaf_h2", "leaf_l1", "leaf_l2"])
        self.assertTrue(any(ev.get("event") == "root_top_m_plan" for ev in trace_top_m.event_log))
        self.assertTrue(
            any(
                ev.get("event") == "reject_leaf_branch_cap" and ev.get("cap_source") == "top_m_budget"
                for ev in trace_top_m.event_log
            )
        )


if __name__ == "__main__":
    unittest.main()
