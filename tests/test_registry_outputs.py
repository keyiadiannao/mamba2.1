from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.tracing import append_jsonl, build_registry_row


class RegistryOutputsTest(unittest.TestCase):
    def test_build_registry_row_extracts_core_fields(self) -> None:
        payload = {
            "run_id": "phase_a_demo_x",
            "question": "What did Einstein propose?",
            "tree_path": "data/processed/demo_tree_payload.json",
            "output_run_dir": "outputs/runs/phase_a_demo_x",
            "config": {
                "navigator_type": "mamba_ssm",
                "navigator_model_name": "mamba2",
                "generator_type": "qwen",
                "generator_model_name": "qwen",
                "routing_mode": "rule",
                "context_source": "t1_visited_leaves_ordered",
            },
            "trace": {
                "routing_mode": "rule",
                "context_source": "t1_visited_leaves_ordered",
                "nav_success": True,
                "rollback_count": 2,
                "snapshot_stack_max_depth": 3,
                "nav_wall_time_ms": 12.5,
                "visited_leaf_indices_deduped": [1, 2],
                "evidence_texts": ["Einstein proposed relativity."],
                "exact_match": 1,
                "rouge_l_f1": None,
            },
        }

        row = build_registry_row(payload)
        self.assertEqual(row["run_id"], "phase_a_demo_x")
        self.assertEqual(row["navigator_type"], "mamba_ssm")
        self.assertEqual(row["navigator_model_name"], "mamba2")
        self.assertEqual(row["generator_type"], "qwen")
        self.assertEqual(row["generator_model_name"], "qwen")
        self.assertEqual(row["visited_leaf_count"], 2)
        self.assertEqual(row["evidence_count"], 1)
        self.assertEqual(row["exact_match"], 1)

    def test_append_jsonl_writes_line_delimited_registry(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp_dir, True)

        registry_path = append_jsonl(temp_dir / "run_registry.jsonl", {"run_id": "x1", "ok": True})
        content = registry_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(content), 1)
        self.assertEqual(json.loads(content[0])["run_id"], "x1")


if __name__ == "__main__":
    unittest.main()
