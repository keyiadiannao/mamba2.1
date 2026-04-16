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


if __name__ == "__main__":
    unittest.main()
