from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.tracing import write_run_payload
from src.tree_builder import load_tree_from_json, load_tree_payload


class PhaseAIOTest(unittest.TestCase):
    def test_tree_payload_loader_reads_question_and_root(self) -> None:
        payload_path = Path("data/processed/demo_tree_payload.json")
        payload = load_tree_payload(payload_path)
        tree = load_tree_from_json(payload_path)

        self.assertEqual(payload["question"], "What did Einstein propose in relativity?")
        self.assertEqual(tree.root.node_id, "root")
        self.assertEqual(len(tree.root.children), 2)

    def test_write_run_payload_creates_run_artifact(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp_dir, True)

        output_path = write_run_payload(
            temp_dir,
            payload={"ok": True, "kind": "phase_a_smoke"},
            run_id="unit_test_run",
        )

        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.parent.name, "unit_test_run")


if __name__ == "__main__":
    unittest.main()
