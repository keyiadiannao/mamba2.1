from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.diagnostics.summarize_never_visit_nav_signals import main as nv_main


class NeverVisitNavSignalsTest(unittest.TestCase):
    def test_summarize_never_visit_counts_max_nodes_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "x"
            run_dir.mkdir(parents=True)
            payload = {
                "run_id": "r1",
                "sample_id": "s1",
                "batch_id": "b1",
                "trace": {
                    "leaf_indices_required": [0],
                    "visited_leaf_indices_deduped": [],
                    "visited_node_ids": ["a"] * 80,
                    "evidence_texts": [],
                    "rollback_count": 1,
                    "snapshot_stack_max_depth": 2,
                    "event_log": [
                        {"event": "max_nodes_reached", "node_id": "n", "depth": 3},
                    ],
                },
            }
            run_path = run_dir / "run_payload.json"
            run_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            audit = {
                "per_sample": [
                    {
                        "n_gold_leaves": 1,
                        "gold_hit_visited": False,
                        "source_path": str(run_path),
                    }
                ]
            }
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")

            rc = nv_main(["--audit-json", str(audit_path), "--root", str(root)])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
