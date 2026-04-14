from __future__ import annotations

import unittest

from src.controller import ControllerConfig, SSGSController
from src.generator_bridge import build_generator_prompt
from src.navigator import MockMambaNavigator
from src.router import RuleRouter
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
            "rollback_count",
            "snapshot_stack_max_depth",
            "nav_wall_time_ms",
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
        self.assertGreaterEqual(len(trace.event_log), 1)
        self.assertGreaterEqual(len(trace.route_decisions), 1)
        self.assertIn("leaf_relativity_1", trace.evidence_node_ids)


if __name__ == "__main__":
    unittest.main()
