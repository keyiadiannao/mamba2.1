from __future__ import annotations

import unittest

from src.tracing import build_navigation_summary, build_registry_row


class NavigationSummaryFieldsTest(unittest.TestCase):
    def test_sample_id_propagates_into_navigation_summary_and_registry(self) -> None:
        payload = {
            "run_id": "run_x",
            "sample_id": "sample_x",
            "question": "q",
            "tree_path": "data/processed/demo_tree_payload.json",
            "output_run_dir": "outputs/runs/run_x",
            "config": {
                "navigator_type": "mamba_ssm",
                "navigator_model_name": "mamba2-smoke",
                "generator_type": "qwen",
                "generator_model_name": "qwen",
                "routing_mode": "cosine_probe",
                "context_source": "t1_visited_leaves_ordered",
            },
            "trace": {
                "routing_mode": "cosine_probe",
                "context_source": "t1_visited_leaves_ordered",
                "nav_success": True,
                "rollback_count": 1,
                "snapshot_stack_max_depth": 2,
                "nav_wall_time_ms": 10.0,
                "visited_leaf_indices_deduped": [1],
                "evidence_texts": ["text"],
                "evidence_node_ids": ["leaf_x"],
                "rouge_l_f1": None,
                "exact_match": 1,
            },
        }
        navigation_summary = build_navigation_summary(payload)
        registry_row = build_registry_row(payload)
        self.assertEqual(navigation_summary["sample_id"], "sample_x")
        self.assertEqual(registry_row["sample_id"], "sample_x")


if __name__ == "__main__":
    unittest.main()
