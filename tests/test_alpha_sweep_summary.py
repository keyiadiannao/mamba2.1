from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_eval.summarize_alpha_sweep import _extract_alpha, _load_jsonl


class AlphaSweepSummaryTest(unittest.TestCase):
    def test_extract_alpha_prefers_config_value(self) -> None:
        alpha = _extract_alpha(
            batch_id="end_to_end_real_corpus_370m_qwen7b_rule_anticollapse_entityalpha_0_3_20260416_123000",
            row={"config": {"entity_boost_alpha": 0.5}},
            prefix="end_to_end_real_corpus_370m_qwen7b_rule_anticollapse_entityalpha_",
        )
        self.assertEqual(alpha, 0.5)

    def test_extract_alpha_from_batch_id_suffix(self) -> None:
        alpha = _extract_alpha(
            batch_id="end_to_end_real_corpus_370m_qwen7b_rule_anticollapse_entityalpha_0_3_20260416_123000",
            row={},
            prefix="end_to_end_real_corpus_370m_qwen7b_rule_anticollapse_entityalpha_",
        )
        self.assertEqual(alpha, 0.3)

    def test_load_jsonl_reads_non_empty_lines(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))
        file_path = temp_dir / "batch_summary.jsonl"
        file_path.write_text(
            json.dumps({"batch_id": "b1"}) + "\n\n" + json.dumps({"batch_id": "b2"}) + "\n",
            encoding="utf-8",
        )
        rows = _load_jsonl(file_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["batch_id"], "b1")
        self.assertEqual(rows[1]["batch_id"], "b2")


if __name__ == "__main__":
    unittest.main()
