from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.diagnostics.analyze_evidence_saturation import (
    _summarize,
    analyze_payload,
    main,
)


class EvidenceSaturationDiagnosticsTest(unittest.TestCase):
    def test_analyze_payload_saturation_and_gold(self) -> None:
        payload = {
            "run_id": "r1",
            "sample_id": "s1",
            "batch_id": "b1",
            "config": {"max_evidence": 3},
            "trace": {
                "evidence_texts": ["a", "b", "c"],
                "evidence_node_ids": [
                    "leaf_doc1__sent_000",
                    "leaf_doc1__sent_001",
                    "leaf_doc2__sent_000",
                ],
                "leaf_indices_required": [2, 5],
                "visited_leaf_indices_deduped": [0, 2, 7],
                "visited_leaf_visits_ordered": [0, 2, 7, 2],
                "event_log": [
                    {"event": "accept_evidence", "node_id": "leaf_x", "leaf_index": 2},
                ],
            },
        }
        row = analyze_payload(payload)
        self.assertTrue(row["saturated"])
        self.assertEqual(row["n_evidence"], 3)
        self.assertEqual(row["unique_entities_in_evidence"], 2)
        self.assertEqual(row["evidence_count_same_entity_as_first"], 2)
        self.assertTrue(row["gold_hit_visited"])
        self.assertEqual(row["gold_index_first_in_visits"], 1)
        self.assertTrue(row["gold_in_accepted_evidence"])
        self.assertEqual(row["n_generator_context_items"], 0)

    def test_generator_context_gold_with_leaf_map(self) -> None:
        payload = {
            "config": {"context_source": "t1_visited_leaves_ordered"},
            "generator_evidence_texts": ["First sentence about alpha."],
            "trace": {
                "context_source": "t1_visited_leaves_ordered",
                "leaf_indices_required": [0],
                "evidence_texts": [],
                "evidence_node_ids": [],
                "visited_leaf_indices_deduped": [],
                "visited_leaf_visits_ordered": [],
                "event_log": [],
            },
        }
        row = analyze_payload(
            payload,
            leaf_index_to_text={0: "First sentence about alpha."},
        )
        self.assertTrue(row["context_gold_metrics_available"])
        self.assertEqual(row["frac_gold_leaf_texts_in_generator_context"], 1.0)
        self.assertTrue(row["all_gold_texts_in_generator_context"])

    def test_summarize_fractions(self) -> None:
        rows = [
            {
                "n_gold_leaves": 1,
                "saturated": True,
                "gold_hit_visited": False,
                "gold_in_accepted_evidence": False,
                "n_evidence": 3,
                "unique_entities_in_evidence": 1,
                "evidence_count_same_entity_as_first": 3,
            },
            {
                "n_gold_leaves": 1,
                "saturated": False,
                "gold_hit_visited": True,
                "gold_in_accepted_evidence": True,
                "n_evidence": 1,
                "unique_entities_in_evidence": 1,
                "evidence_count_same_entity_as_first": 1,
            },
        ]
        s = _summarize(rows)
        self.assertEqual(s["sample_count"], 2)
        self.assertEqual(s["frac_evidence_budget_saturated"], 0.5)
        self.assertEqual(s["frac_gold_leaf_ever_visited_deduped"], 0.5)
        self.assertEqual(s["frac_gold_in_accepted_evidence"], 0.5)
        self.assertEqual(s["frac_saturated_among_gold_missing"], 1.0)

    def test_cli_registry_smoke(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))

        run_dir = temp_dir / "outputs" / "runs" / "run_a"
        run_dir.mkdir(parents=True)
        payload = {
            "run_id": "run_a",
            "sample_id": "s1",
            "batch_id": "batch_x",
            "config": {"max_evidence": 2},
            "trace": {
                "evidence_texts": ["x", "y"],
                "evidence_node_ids": ["n1", "n2"],
                "leaf_indices_required": [],
                "visited_leaf_indices_deduped": [],
                "visited_leaf_visits_ordered": [],
                "event_log": [],
            },
        }
        (run_dir / "run_payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        reg_path = temp_dir / "outputs" / "reports" / "run_registry.jsonl"
        reg_path.parent.mkdir(parents=True)
        row = {
            "run_id": "run_a",
            "batch_id": "batch_x",
            "output_run_dir": str(run_dir),
        }
        reg_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        out_json = temp_dir / "out" / "rep.json"
        code = main(
            [
                "--registry-jsonl",
                str(reg_path.relative_to(temp_dir)),
                "--batch-id",
                "batch_x",
                "--root",
                str(temp_dir),
                "--out-json",
                str(out_json.relative_to(temp_dir)),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(out_json.exists())

    def test_list_batch_ids_runs(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))

        reg_path = temp_dir / "run_registry.jsonl"
        reg_path.write_text(
            json.dumps({"batch_id": "batch_a", "output_run_dir": "/tmp/x"}, ensure_ascii=False) + "\n"
            + json.dumps({"batch_id": "batch_b", "output_run_dir": "/tmp/y"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        code = main(
            [
                "--registry-jsonl",
                str(reg_path.relative_to(temp_dir)),
                "--root",
                str(temp_dir),
                "--list-batch-ids",
            ]
        )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
