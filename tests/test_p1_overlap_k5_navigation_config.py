"""P1-2: overlap k=5 differs from P0-2 rule reg200 only by context_select_k and id prefixes."""

from __future__ import annotations

import json
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestP1OverlapK5NavigationConfig(unittest.TestCase):
    def test_k5_vs_p0_rule_reg200(self) -> None:
        p0 = json.loads(
            (
                ROOT / "configs/experiment/navigation_batch_real_corpus_p0_frozen_nav_reg200_rule.example.json"
            ).read_text(encoding="utf-8")
        )
        p1 = json.loads(
            (
                ROOT / "configs/experiment/navigation_batch_real_corpus_p1_rule_frozen_nav_reg200_overlap_k5.example.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(p0["context_select_k"], 4)
        self.assertEqual(p1["context_select_k"], 5)

        label = {"batch_id_prefix", "run_id_prefix", "context_select_k"}
        for k, v in p0.items():
            if k in label:
                continue
            self.assertEqual(p1[k], v, msg=f"unexpected mismatch on {k!r}")

        self.assertEqual(p1["batch_id_prefix"], "nav_p1_reg200_rule_overlap_k5")


if __name__ == "__main__":
    unittest.main()
