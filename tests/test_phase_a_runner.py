from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline import build_batch_summary, load_json


class PhaseARunnerTest(unittest.TestCase):
    def test_load_json_reads_batch_samples(self) -> None:
        payload = load_json(Path("data/processed/demo_navigation_batch.json"))
        self.assertEqual(len(payload["samples"]), 3)

    def test_build_batch_summary_aggregates_navigation_metrics(self) -> None:
        summary = build_batch_summary(
            "batch_x",
            [
                {"run_id": "r1", "sample_id": "s1", "trace": {"nav_success": True, "exact_match": 1, "nav_wall_time_ms": 10}},
                {"run_id": "r2", "sample_id": "s2", "trace": {"nav_success": False, "exact_match": None, "nav_wall_time_ms": 30}},
            ],
        )
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["nav_success_count"], 1)
        self.assertEqual(summary["exact_match_count"], 1)
        self.assertEqual(summary["avg_nav_wall_time_ms"], 20.0)

    def test_batch_runner_reuses_single_controller_instance(self) -> None:
        config_path = Path("configs/experiment/navigation_batch_demo.json").resolve()
        call_counter = {"count": 0}

        def fake_build_controller(config):
            call_counter["count"] += 1
            return object()

        with patch("src.pipeline.build_controller", side_effect=fake_build_controller), patch(
            "src.pipeline.run_navigation_sample"
        ) as mock_run_navigation_sample, patch(
            "src.tracing.make_run_id",
            return_value="batch_test",
        ), patch("src.tracing.write_json"), patch("src.tracing.append_jsonl"):
            mock_run_navigation_sample.side_effect = [
                {"run_id": "r1", "sample_id": "s1", "trace": {"nav_success": True, "exact_match": 1, "nav_wall_time_ms": 10}},
                {"run_id": "r2", "sample_id": "s2", "trace": {"nav_success": True, "exact_match": 0, "nav_wall_time_ms": 20}},
                {"run_id": "r3", "sample_id": "s3", "trace": {"nav_success": False, "exact_match": None, "nav_wall_time_ms": 30}},
            ]

            with patch(
                "sys.argv",
                ["run_navigation_batch.py", "--config", str(config_path)],
            ):
                runpy.run_module("scripts.run_nav.run_navigation_batch", run_name="__main__")

        self.assertEqual(call_counter["count"], 1)
        self.assertEqual(mock_run_navigation_sample.call_count, 3)
        shared_controller = mock_run_navigation_sample.call_args_list[0].kwargs["controller"]
        for call in mock_run_navigation_sample.call_args_list[1:]:
            self.assertIs(call.kwargs["controller"], shared_controller)


if __name__ == "__main__":
    unittest.main()
