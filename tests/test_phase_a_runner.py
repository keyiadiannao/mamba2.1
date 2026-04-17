from __future__ import annotations

import runpy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.pipeline import build_batch_summary, load_json, run_navigation_sample
from src.pipeline.phase_a_runner import (
    _apply_evidence_controls,
    _context_build_max_items,
    _postprocess_generated_answer,
)


class PhaseARunnerTest(unittest.TestCase):
    def test_postprocess_constrained_projects_answer_to_binary_options(self) -> None:
        answer, mode, rule = _postprocess_generated_answer(
            question="Which film was released more recently, Royal Treasure or When Love Begins?",
            generated_answer="The more recent one is Royal Treasure.",
            config={"postprocess_mode": "constrained"},
        )
        self.assertEqual(answer, "Royal Treasure")
        self.assertEqual(mode, "constrained")
        self.assertEqual(rule, "force_binary_choice")

    def test_apply_evidence_controls_anti_collapse_limits_entity_and_requires_overlap(self) -> None:
        class _Node:
            def __init__(self, node_id: str, text: str) -> None:
                self.node_id = node_id
                self.text = text

        selected_nodes = [
            _Node("leaf_sergei_parajanov__sent_000_000_000", "Sergei Parajanov died in 1990."),
            _Node("leaf_sergei_parajanov__sent_000_000_001", "He was a Soviet artist."),
            _Node("leaf_royal_treasure__sent_000_000_000", "Royal Treasure was released in 2016."),
            _Node("leaf_when_love_begins__sent_000_000_000", "When Love Begins is a 2008 film."),
        ]
        filtered = _apply_evidence_controls(
            selected_nodes=selected_nodes,
            question="Which film was released more recently, Royal Treasure or When Love Begins?",
            config={
                "evidence_control_mode": "anti_collapse",
                "evidence_control_per_entity_max": 1,
                "evidence_control_require_question_overlap": True,
            },
        )
        self.assertEqual([node.node_id for node in filtered], [
            "leaf_royal_treasure__sent_000_000_000",
            "leaf_when_love_begins__sent_000_000_000",
        ])

    def test_run_navigation_sample_writes_entity_metrics(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))

        tree_dir = temp_dir / "data" / "processed"
        tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = tree_dir / "entity_tree_payload.json"
        tree_path.write_text(
            json.dumps(
                {
                    "question": "What did Einstein discover?",
                    "reference_answer": "relativity",
                    "root": {
                        "node_id": "root",
                        "text": "science index",
                        "children": [
                            {"node_id": "leaf_einstein", "text": "Einstein discovered relativity.", "leaf_index": 0},
                            {"node_id": "leaf_newton", "text": "Newton studied gravity.", "leaf_index": 1},
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
                "run_generator": False,
                "entity_boost_alpha": 0.3,
            },
            question="What did Einstein discover?",
            tree_path="data/processed/entity_tree_payload.json",
            reference_answer="relativity",
            run_id_prefix="test_entity_metrics",
            sample_id="sample_entity_metrics",
        )
        trace = payload["trace"]
        self.assertEqual(trace["entity_boost_alpha"], 0.3)
        self.assertEqual(trace["question_entity_count"], 1)
        self.assertEqual(trace["entity_intersection_size"], 1)
        self.assertEqual(trace["entity_hit_rate"], 1.0)

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

    def test_run_navigation_sample_context_select_first_k(self) -> None:
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
                            {"node_id": "leaf_c", "text": "gamma", "leaf_index": 2},
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
                "context_max_items": 3,
                "context_select_mode": "first_k",
                "context_select_k": 2,
                "run_generator": False,
            },
            question="q",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="a",
            run_id_prefix="test_context_select_first_k",
            sample_id="sample_context_select_first_k",
        )

        self.assertEqual(payload["generator_evidence_texts"], ["alpha", "beta"])
        self.assertEqual(payload["generator_evidence_node_ids"], ["leaf_a", "leaf_b"])

    def test_run_navigation_sample_context_select_dedupe_entity_then_k(self) -> None:
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
                            {
                                "node_id": "leaf_einstein__sent_000_000_000",
                                "text": "Einstein alpha.",
                                "leaf_index": 0,
                            },
                            {
                                "node_id": "leaf_einstein__sent_000_000_001",
                                "text": "Einstein beta.",
                                "leaf_index": 1,
                            },
                            {
                                "node_id": "leaf_newton__sent_000_000_000",
                                "text": "Newton gamma.",
                                "leaf_index": 2,
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
                "context_source": "flat_leaf_concat",
                "context_max_items": 3,
                "context_select_mode": "dedupe_entity_then_k",
                "context_select_k": 2,
                "run_generator": False,
            },
            question="q",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="a",
            run_id_prefix="test_context_select_dedupe",
            sample_id="sample_context_select_dedupe",
        )

        self.assertEqual(
            payload["generator_evidence_node_ids"],
            ["leaf_einstein__sent_000_000_000", "leaf_newton__sent_000_000_000"],
        )

    def test_run_navigation_sample_context_select_question_overlap_topk(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))

        tree_dir = temp_dir / "data" / "processed"
        tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = tree_dir / "demo_tree_payload.json"
        tree_path.write_text(
            json.dumps(
                {
                    "question": "Which scientist developed relativity and gravity insights?",
                    "reference_answer": "Einstein",
                    "root": {
                        "node_id": "root",
                        "text": "root",
                        "children": [
                            {"node_id": "leaf_c", "text": "General science notes.", "leaf_index": 0},
                            {
                                "node_id": "leaf_a",
                                "text": "Einstein developed relativity and studied gravity.",
                                "leaf_index": 1,
                            },
                            {"node_id": "leaf_b", "text": "Newton studied gravity and motion.", "leaf_index": 2},
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
                "context_max_items": 3,
                "context_select_mode": "question_overlap_topk",
                "context_select_k": 2,
                "run_generator": False,
            },
            question="Which scientist developed relativity and gravity insights?",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="Einstein",
            run_id_prefix="test_context_select_overlap",
            sample_id="sample_context_select_overlap",
        )

        self.assertEqual(payload["generator_evidence_node_ids"], ["leaf_a", "leaf_b"])

    def test_run_navigation_sample_context_select_question_entity_match_topk(self) -> None:
        """Entity-first ranking can differ from token overlap when generic leaves share many words."""
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))

        tree_dir = temp_dir / "data" / "processed"
        tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = tree_dir / "demo_tree_payload.json"
        tree_path.write_text(
            json.dumps(
                {
                    "question": "Any news on \"ZebraCorp\" guidance and quarterly updates?",
                    "reference_answer": "ZebraCorp",
                    "root": {
                        "node_id": "root",
                        "text": "root",
                        "children": [
                            {
                                "node_id": "leaf_generic",
                                "text": (
                                    "Guidance news quarterly updates disclosures commentary trends "
                                    "markets corporate investor sector standards reporting filings."
                                ),
                                "leaf_index": 0,
                            },
                            {
                                "node_id": "leaf_target",
                                "text": "ZebraCorp statement.",
                                "leaf_index": 1,
                            },
                            {
                                "node_id": "leaf_other",
                                "text": "Unrelated macro notes.",
                                "leaf_index": 2,
                            },
                        ],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        base_cfg: dict = {
            "output_dir": "outputs/runs",
            "navigator_type": "mock",
            "routing_mode": "rule",
            "context_source": "flat_leaf_concat",
            "context_max_items": 3,
            "context_select_k": 1,
            "run_generator": False,
        }

        payload_overlap = run_navigation_sample(
            root_dir=temp_dir,
            config={**base_cfg, "context_select_mode": "question_overlap_topk"},
            question="Any news on \"ZebraCorp\" guidance and quarterly updates?",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="ZebraCorp",
            run_id_prefix="test_ctx_entity_overlap",
            sample_id="sample_ctx_entity_overlap",
        )
        payload_entity = run_navigation_sample(
            root_dir=temp_dir,
            config={**base_cfg, "context_select_mode": "question_entity_match_topk"},
            question="Any news on \"ZebraCorp\" guidance and quarterly updates?",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="ZebraCorp",
            run_id_prefix="test_ctx_entity_entity",
            sample_id="sample_ctx_entity_entity",
        )

        self.assertEqual(payload_overlap["generator_evidence_node_ids"][0], "leaf_generic")
        self.assertEqual(payload_entity["generator_evidence_node_ids"][0], "leaf_target")

    def test_context_build_max_items_respects_pool_only_when_select_active(self) -> None:
        self.assertEqual(
            _context_build_max_items({"context_select_mode": "off", "context_select_pool_max_items": 99}, 8),
            8,
        )
        self.assertEqual(
            _context_build_max_items({"context_select_mode": "question_overlap_topk"}, 8),
            8,
        )
        self.assertEqual(
            _context_build_max_items(
                {"context_select_mode": "question_overlap_topk", "context_select_pool_max_items": 20},
                8,
            ),
            20,
        )

    def test_context_select_pool_max_items_widen_overlap_candidates(self) -> None:
        """Larger pool feeds more leaves into overlap ranking before top-k."""
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(temp_dir, ignore_errors=True))

        children = []
        for i in range(10):
            if i < 3:
                text = f"Misc filler paragraph {i} unrelated travel notes."
            else:
                text = f"Physics block {i}: Einstein relativity and gravity concepts."
            children.append({"node_id": f"leaf_{i}", "text": text, "leaf_index": i})

        tree_dir = temp_dir / "data" / "processed"
        tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = tree_dir / "demo_tree_payload.json"
        tree_path.write_text(
            json.dumps(
                {
                    "question": "What about Einstein relativity and gravity?",
                    "reference_answer": "Einstein",
                    "root": {"node_id": "root", "text": "root", "children": children},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        base_cfg: dict = {
            "output_dir": "outputs/runs",
            "navigator_type": "mock",
            "routing_mode": "rule",
            "context_source": "flat_leaf_concat",
            "context_max_items": 3,
            "context_select_mode": "question_overlap_topk",
            "context_select_k": 2,
            "run_generator": False,
        }

        payload_narrow = run_navigation_sample(
            root_dir=temp_dir,
            config=dict(base_cfg),
            question="What about Einstein relativity and gravity?",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="Einstein",
            run_id_prefix="test_ctx_pool_narrow",
            sample_id="sample_ctx_pool_narrow",
        )
        payload_wide = run_navigation_sample(
            root_dir=temp_dir,
            config={**base_cfg, "context_select_pool_max_items": 10},
            question="What about Einstein relativity and gravity?",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="Einstein",
            run_id_prefix="test_ctx_pool_wide",
            sample_id="sample_ctx_pool_wide",
        )

        self.assertEqual(payload_narrow["generator_evidence_node_ids"], ["leaf_0", "leaf_1"])
        self.assertEqual(payload_wide["generator_evidence_node_ids"], ["leaf_3", "leaf_4"])

    def test_context_build_failure_scores_zero_instead_of_falling_back(self) -> None:
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
                "context_source": "oracle_item_leaves",
                "run_generator": False,
            },
            question="What did Einstein propose in relativity?",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="Einstein proposed relativity, including special relativity and general relativity.",
            run_id_prefix="test_context_failure",
            sample_id="sample_context_failure",
        )

        self.assertIn("oracle_item_leaves", payload["trace"]["context_build_error"])
        self.assertEqual(payload["trace"]["exact_match"], 0)
        self.assertEqual(payload["trace"]["answer_f1"], 0.0)
        self.assertEqual(payload["trace"]["rouge_l_f1"], 0.0)
        self.assertEqual(payload["trace"]["generation_error"], "skipped_due_to_context_error")
        self.assertIsNone(payload["generator_prompt"])
        self.assertEqual(payload["trace"]["failure_attribution"], "context_construction_failure")
        self.assertEqual(payload["eval_mode"], "retrieval")

    def test_context_build_failure_skips_generator_when_run_generator_true(self) -> None:
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

        with patch("src.generator_bridge.runner.generate_answer") as mock_generate:
            payload = run_navigation_sample(
                root_dir=temp_dir,
                config={
                    "output_dir": "outputs/runs",
                    "navigator_type": "mock",
                    "routing_mode": "rule",
                    "context_source": "oracle_item_leaves",
                    "run_generator": True,
                    "generator_type": "mock",
                    "generator_inference_mode": "extractive_first_evidence",
                },
                question="What did Einstein propose in relativity?",
                tree_path="data/processed/demo_tree_payload.json",
                reference_answer="Einstein proposed relativity, including special relativity and general relativity.",
                run_id_prefix="test_context_skip_gen",
                sample_id="sample_context_skip_gen",
            )

        mock_generate.assert_not_called()
        self.assertEqual(payload["trace"]["generation_error"], "skipped_due_to_context_error")
        self.assertEqual(payload["eval_mode"], "generation")

    def test_generation_failure_scores_zero_in_end_to_end_mode(self) -> None:
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
                "generator_type": "qwen",
                "generator_inference_mode": "hf_causal_lm",
                "generator_model_name": "qwen",
            },
            question="What did Einstein propose in relativity?",
            tree_path="data/processed/demo_tree_payload.json",
            reference_answer="Einstein proposed relativity, including special relativity and general relativity.",
            run_id_prefix="test_generation_failure",
            sample_id="sample_generation_failure",
        )

        self.assertIsNotNone(payload["trace"]["generation_error"])
        self.assertEqual(payload["trace"]["exact_match"], 0)
        self.assertEqual(payload["trace"]["answer_f1"], 0.0)
        self.assertEqual(payload["trace"]["rouge_l_f1"], 0.0)

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
