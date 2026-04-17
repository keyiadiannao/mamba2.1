from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class EndToEndBatchMaxSamplesTest(unittest.TestCase):
    def test_max_samples_truncates(self) -> None:
        config_path = ROOT / "configs/experiment/end_to_end_batch_demo_smoke_ctxsel_overlap_k4.json"

        def fake_build_controller(config):
            return object()

        with patch("src.pipeline.build_controller", side_effect=fake_build_controller), patch(
            "src.pipeline.run_navigation_sample"
        ) as mock_run, patch("src.tracing.make_run_id", return_value="e2e_max2"), patch(
            "src.tracing.write_json"
        ) as mock_write, patch("src.tracing.append_jsonl"):
            mock_run.side_effect = [
                {"run_id": "r1", "sample_id": "s1", "trace": {"nav_success": True, "exact_match": 1}},
                {"run_id": "r2", "sample_id": "s2", "trace": {"nav_success": True, "exact_match": 0}},
            ]
            with patch(
                "sys.argv",
                [
                    "run_end_to_end_batch.py",
                    "--config",
                    str(config_path),
                    "--max-samples",
                    "2",
                ],
            ):
                runpy.run_path(str(ROOT / "scripts/run_eval/run_end_to_end_batch.py"), run_name="__main__")

        self.assertEqual(mock_run.call_count, 2)
        written = mock_write.call_args[0][1]
        self.assertEqual(written["manifest_sample_count"], 3)
        self.assertEqual(written["max_samples"], 2)
        self.assertEqual(written["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
