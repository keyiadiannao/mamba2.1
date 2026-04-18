"""Oracle-item-leaves nav batch: ceiling context; manifest must carry positive_leaf_indices."""

from __future__ import annotations

import json
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestNavReg200OracleConfig(unittest.TestCase):
    def test_oracle_context_keys(self) -> None:
        cfg = json.loads(
            (
                ROOT / "configs/experiment/navigation_batch_real_corpus_nav_reg200_oracle_item_leaves.example.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(cfg["context_source"], "oracle_item_leaves")
        self.assertEqual(str(cfg.get("context_select_mode", "")).lower(), "off")


if __name__ == "__main__":
    unittest.main()
