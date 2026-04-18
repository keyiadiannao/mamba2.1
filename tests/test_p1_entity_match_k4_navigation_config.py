"""P1-3: entity_match mode differs from P0-2 rule overlap only by context_select_mode and id prefixes."""

from __future__ import annotations

import json
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestP1EntityMatchK4NavigationConfig(unittest.TestCase):
    def test_entity_match_vs_p0_rule_reg200(self) -> None:
        p0 = json.loads(
            (
                ROOT / "configs/experiment/navigation_batch_real_corpus_p0_frozen_nav_reg200_rule.example.json"
            ).read_text(encoding="utf-8")
        )
        p1 = json.loads(
            (
                ROOT / "configs/experiment/navigation_batch_real_corpus_p1_rule_frozen_nav_reg200_entity_match_k4.example.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(p0["context_select_mode"], "question_overlap_topk")
        self.assertEqual(p1["context_select_mode"], "question_entity_match_topk")

        label = {"batch_id_prefix", "run_id_prefix", "context_select_mode"}
        for k, v in p0.items():
            if k in label:
                continue
            self.assertEqual(p1[k], v, msg=f"unexpected mismatch on {k!r}")


if __name__ == "__main__":
    unittest.main()
