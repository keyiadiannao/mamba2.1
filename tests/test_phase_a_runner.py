from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
