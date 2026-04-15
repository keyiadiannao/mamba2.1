from __future__ import annotations

import runpy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.pipeline import build_batch_summary, load_json, run_navigation_sample


class PhaseARunnerTest(unittest.TestCase):
    def test_load_json_reads_batch_samples(self) -> None:
        payload = load_json(Path("data/processed/demo_navigation_batch.json"))
        self.assertEqual(len(payload["samples"]), 3)

    def test_build_batch_summary_aggregates_navigation_metrics(self) -> None:
        summary = build_batch_summary(
            "batch_x",
            [
                {
                    "run_id": "r1",
                    "sample_id": "s1",
                    "trace": {"nav_success": True, "exact_match": 1, "answer_f1": 1.0, "rouge_l_f1": 1.0, "nav_wall_time_ms": 10},
                },
                {
                    "run_id": "r2",
                    "sample_id": "s2",
                    "trace": {"nav_success": False, "exact_match": None, "answer_f1": None, "rouge_l_f1": None, "nav_wall_time_ms": 30},
                },
            ],
        )
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["nav_success_count"], 1)
        self.assertEqual(summary["exact_match_count"], 1)
        self.assertEqual(summary["avg_nav_wall_time_ms"], 20.0)
        self.assertEqual(summary["avg_answer_f1"], 1.0)
        self.assertEqual(summary["avg_rouge_l_f1"], 1.0)

    def test_run_navigation_sample_can_score_generated_answer_with_mock_generator(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))

        tree_dir = temp_dir / "data" / "processed"
        tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = tree_dir / "demo_tree_payload.json"
        tree_path.write_text(
            json.dumps(
                {
                    "question": "What did Einstein propose in relativity?",
                    "reference_answer": "Einstein proposed relativity, including special relativity and general relativity.",
                    "root": {
                        "node_id": "root",
                        "text": "physics knowledge index",
                        "children": [
                            {
                                "node_id": "branch_relativity",
                                "text": "Einstein relativity branch",
                                "children": [
                                    {
                                        "node_id": "leaf_relativity_1",
                                        "text": "Einstein proposed relativity, including special relativity and general relativity.",
                                        "leaf_index": 0,
                                    }
                                ],
                            }
                        ],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        payload = run_navigation_sample(
            root_dir=temp_dir,
            config={
                "output_dir": "outputs/runs",
                "navigator_type": "mock",
                "routing_mode": "rule",
                "context_source": "t1_visited_leaves_ordered",
                "run_generator": True,
                "generator_type": "mock",
                "generator_inference_mode": "extractive_first_evidence",
            },
            question="What did Einstein propose in relativity?",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="Einstein proposed relativity, including special relativity and general relativity.",
            run_id_prefix="test_e2e",
            sample_id="sample_x",
        )

        self.assertEqual(
            payload["generated_answer"],
            "Einstein proposed relativity, including special relativity and general relativity.",
        )
        self.assertEqual(payload["trace"]["exact_match"], 1)
        self.assertEqual(payload["trace"]["answer_f1"], 1.0)
        self.assertEqual(payload["trace"]["rouge_l_f1"], 1.0)

    def test_run_navigation_sample_supports_oracle_context_source(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))

        tree_dir = temp_dir / "data" / "processed"
        tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = tree_dir / "demo_tree_payload.json"
        tree_path.write_text(
            json.dumps(
                {
                    "question": "What did Einstein propose in relativity?",
                    "reference_answer": "Einstein proposed relativity, including special relativity and general relativity.",
                    "root": {
                        "node_id": "root",
                        "text": "physics knowledge index",
                        "children": [
                            {
                                "node_id": "branch_newton",
                                "text": "Newton branch",
                                "children": [
                                    {
                                        "node_id": "leaf_newton_1",
                                        "text": "Newtonian mechanics explains force, motion, and gravity.",
                                        "leaf_index": 0,
                                    }
                                ],
                            },
                            {
                                "node_id": "branch_relativity",
                                "text": "Einstein relativity branch",
                                "children": [
                                    {
                                        "node_id": "leaf_relativity_1",
                                        "text": "Einstein proposed relativity, including special relativity and general relativity.",
                                        "leaf_index": 1,
                                    }
                                ],
                            },
                        ],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        payload = run_navigation_sample(
            root_dir=temp_dir,
            config={
                "output_dir": "outputs/runs",
                "navigator_type": "mock",
                "routing_mode": "rule",
                "context_source": "oracle_item_leaves",
                "run_generator": False,
            },
            question="What did Einstein propose in relativity?",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="Einstein proposed relativity, including special relativity and general relativity.",
            run_id_prefix="test_oracle_context",
            sample_id="sample_oracle",
            leaf_indices_required=[1],
        )

        self.assertEqual(
            payload["generator_evidence_texts"],
            ["Einstein proposed relativity, including special relativity and general relativity."],
        )
        self.assertEqual(payload["generator_evidence_node_ids"], ["leaf_relativity_1"])

    def test_run_navigation_sample_supports_flat_context_source(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))

        tree_dir = temp_dir / "data" / "processed"
        tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = tree_dir / "demo_tree_payload.json"
        tree_path.write_text(
            json.dumps(
                {
                    "question": "q",
                    "reference_answer": "a",
                    "root": {
                        "node_id": "root",
                        "text": "root",
                        "children": [
                            {"node_id": "leaf_a", "text": "alpha", "leaf_index": 0},
                            {"node_id": "leaf_b", "text": "beta", "leaf_index": 1},
                        ],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        payload = run_navigation_sample(
            root_dir=temp_dir,
            config={
                "output_dir": "outputs/runs",
                "navigator_type": "mock",
                "routing_mode": "rule",
                "context_source": "flat_leaf_concat",
                "context_max_items": 2,
                "run_generator": False,
            },
            question="q",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="a",
            run_id_prefix="test_flat_context",
            sample_id="sample_flat",
        )

        self.assertEqual(payload["generator_evidence_texts"], ["alpha", "beta"])

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
