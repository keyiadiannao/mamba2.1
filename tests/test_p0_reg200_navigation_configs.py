"""P0-2 navigation batch configs: JSON valid and pairwise key alignment (minus routing arm)."""

from __future__ import annotations

import json
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    path = ROOT / "configs" / "experiment" / name
    return json.loads(path.read_text(encoding="utf-8"))


class TestP0Reg200NavigationConfigs(unittest.TestCase):
    def test_configs_load_and_differ_only_by_routing_arm(self) -> None:
        rule_cfg = _load("navigation_batch_real_corpus_p0_frozen_nav_reg200_rule.example.json")
        blend_cfg = _load("navigation_batch_real_corpus_p0_frozen_nav_reg200_learned_root_blend05.example.json")

        self.assertEqual(rule_cfg["routing_mode"], "rule")
        self.assertEqual(blend_cfg["routing_mode"], "learned_root_classifier")
        self.assertEqual(blend_cfg.get("learned_root_blend_alpha"), 0.5)
        self.assertEqual(
            blend_cfg.get("router_checkpoint_path"),
            "configs/router/learned_root_router_real_corpus.json",
        )

        rk = set(rule_cfg.keys())
        bk = set(blend_cfg.keys())
        self.assertEqual(
            bk - rk,
            {"learned_root_blend_alpha", "router_checkpoint_path"},
            msg="Learned arm should only add blend + checkpoint vs rule frozen template.",
        )

        label_keys = {"routing_mode", "batch_id_prefix", "run_id_prefix"}
        common = rk & bk
        for k in common - label_keys:
            self.assertEqual(rule_cfg[k], blend_cfg[k], msg=f"mismatch on key {k!r}")

        self.assertEqual(rule_cfg["context_select_pool_max_items"], 20)
        self.assertEqual(rule_cfg["explore_top_m_root_children"], 0)
        self.assertEqual(rule_cfg["samples_path"], "data/processed/real_corpus_navigation_batch.json")


if __name__ == "__main__":
    unittest.main()
