"""Unit tests for accept_gate_audit (gold visit vs accept_evidence)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.diagnostics.accept_gate_audit import audit_payload, summarize


class TestAcceptGateAudit(unittest.TestCase):
    def test_accepted_gold(self) -> None:
        payload = {
            "run_id": "r1",
            "trace": {
                "leaf_indices_required": [7],
                "visited_leaf_indices_deduped": [7],
                "visited_leaf_visits_ordered": [7],
                "evidence_texts": ["t"],
                "event_log": [
                    {"event": "accept_evidence", "leaf_index": 7, "score": 2.0},
                ],
            },
            "config": {"max_evidence": 8, "min_relevance_score": 1.0},
        }
        row = audit_payload(payload)
        self.assertEqual(row["n_gold_visited_not_accepted_leaves"], 0)
        self.assertTrue(row["all_visited_gold_in_accepted"])

    def test_reject_leaf_below_threshold(self) -> None:
        payload = {
            "trace": {
                "leaf_indices_required": [3],
                "visited_leaf_indices_deduped": [3],
                "visited_leaf_visits_ordered": [3],
                "evidence_texts": [],
                "event_log": [
                    {"event": "reject_leaf", "leaf_index": 3, "score": 0.1},
                ],
            },
            "config": {"max_evidence": 8, "min_relevance_score": 1.0},
        }
        row = audit_payload(payload)
        self.assertEqual(row["n_gold_visited_not_accepted_leaves"], 1)
        self.assertEqual(
            row["visited_not_accepted_dispositions"],
            {"reject_leaf_min_relevance": 1},
        )

    def test_never_visit_gold(self) -> None:
        payload = {
            "trace": {
                "leaf_indices_required": [1, 2],
                "visited_leaf_indices_deduped": [9],
                "visited_leaf_visits_ordered": [9],
                "evidence_texts": [],
                "event_log": [],
            },
            "config": {},
        }
        row = audit_payload(payload)
        self.assertEqual(row["n_gold_never_visited"], 2)
        self.assertFalse(row["gold_hit_visited"])

    def test_summarize_two_samples(self) -> None:
        rows = [
            audit_payload(
                {
                    "trace": {
                        "leaf_indices_required": [1],
                        "visited_leaf_indices_deduped": [1],
                        "visited_leaf_visits_ordered": [1],
                        "evidence_texts": ["x"],
                        "event_log": [{"event": "accept_evidence", "leaf_index": 1}],
                    },
                    "config": {},
                }
            ),
            audit_payload(
                {
                    "trace": {
                        "leaf_indices_required": [2],
                        "visited_leaf_indices_deduped": [2],
                        "visited_leaf_visits_ordered": [2],
                        "evidence_texts": [],
                        "event_log": [{"event": "reject_leaf", "leaf_index": 2, "score": 0.0}],
                    },
                    "config": {},
                }
            ),
        ]
        s = summarize(rows)
        self.assertEqual(s["sample_count_with_gold_annotation"], 2)
        self.assertEqual(s["frac_samples_never_visit_any_gold"], 0.0)
        self.assertEqual(s["frac_samples_visit_gold_but_missing_accept_for_some_visited_gold"], 0.5)

    def test_context_gold_metrics_with_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree = {
                "node_id": "root",
                "text": "r",
                "children": [
                    {
                        "node_id": "gold_leaf",
                        "text": "g",
                        "children": [],
                        "metadata": {"leaf_index": 7},
                    }
                ],
            }
            (root / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
            payload = {
                "tree_path": "tree.json",
                "trace": {
                    "leaf_indices_required": [7],
                    "visited_leaf_indices_deduped": [7],
                    "visited_leaf_visits_ordered": [7],
                    "context_node_ids": ["gold_leaf"],
                    "evidence_texts": [],
                    "event_log": [{"event": "accept_evidence", "leaf_index": 7}],
                },
                "config": {},
            }
            row = audit_payload(payload, root_dir=root)
            self.assertTrue(row["context_gold_metrics_available"])
            self.assertEqual(row["n_gold_leaves_in_context"], 1)
            self.assertEqual(row["frac_gold_leaves_in_context"], 1.0)
            self.assertEqual(row["n_accepted_gold_not_in_context"], 0)
            self.assertEqual(row["n_gold_leaves_accepted"], 1)
            self.assertEqual(row["frac_gold_leaves_accepted"], 1.0)

    def test_accepted_gold_missing_from_context(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree = {
                "node_id": "root",
                "text": "r",
                "children": [
                    {
                        "node_id": "gold_leaf",
                        "text": "g",
                        "children": [],
                        "metadata": {"leaf_index": 3},
                    }
                ],
            }
            (root / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
            payload = {
                "tree_path": "tree.json",
                "trace": {
                    "leaf_indices_required": [3],
                    "visited_leaf_indices_deduped": [3],
                    "visited_leaf_visits_ordered": [3],
                    "context_node_ids": ["other_node"],
                    "evidence_texts": [],
                    "event_log": [{"event": "accept_evidence", "leaf_index": 3}],
                },
                "config": {},
            }
            row = audit_payload(payload, root_dir=root)
            self.assertEqual(row["n_gold_leaves_in_context"], 0)
            self.assertEqual(row["n_accepted_gold_not_in_context"], 1)

    def test_summarize_includes_context_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree = {
                "node_id": "root",
                "text": "r",
                "children": [
                    {
                        "node_id": "L1",
                        "text": "a",
                        "children": [],
                        "metadata": {"leaf_index": 1},
                    }
                ],
            }
            (root / "tree.json").write_text(json.dumps(tree), encoding="utf-8")
            p1 = {
                "tree_path": "tree.json",
                "trace": {
                    "leaf_indices_required": [1],
                    "visited_leaf_indices_deduped": [1],
                    "visited_leaf_visits_ordered": [1],
                    "context_node_ids": ["L1"],
                    "evidence_texts": [],
                    "event_log": [{"event": "accept_evidence", "leaf_index": 1}],
                },
                "config": {},
            }
            p2 = {
                "tree_path": "tree.json",
                "trace": {
                    "leaf_indices_required": [1],
                    "visited_leaf_indices_deduped": [1],
                    "visited_leaf_visits_ordered": [1],
                    "context_node_ids": [],
                    "evidence_texts": [],
                    "event_log": [{"event": "accept_evidence", "leaf_index": 1}],
                },
                "config": {},
            }
            rows = [audit_payload(p1, root_dir=root), audit_payload(p2, root_dir=root)]
            s = summarize(rows)
            self.assertEqual(s["context_gold_metrics_sample_count"], 2)
            self.assertEqual(s["frac_samples_with_any_gold_in_context"], 0.5)
            self.assertEqual(s["sum_accepted_gold_not_in_context"], 1)


if __name__ == "__main__":
    unittest.main()
