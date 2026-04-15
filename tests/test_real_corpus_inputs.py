from __future__ import annotations

import unittest

from src.tree_builder import (
    build_doc_leaf_index_map,
    build_navigation_samples_from_qa,
    build_tree_payload_from_corpus,
)


class RealCorpusInputsTest(unittest.TestCase):
    def test_build_doc_leaf_index_map_collects_leaf_indices(self) -> None:
        tree_payload = build_tree_payload_from_corpus(
            [
                {"doc_id": "doc_a", "title": "Doc A", "text": "alpha beta gamma"},
                {"doc_id": "doc_b", "title": "Doc B", "text": "delta epsilon zeta"},
            ],
            max_chars_per_leaf=200,
        )

        leaf_index_map = build_doc_leaf_index_map(tree_payload)
        self.assertEqual(leaf_index_map["doc_a"], [0])
        self.assertEqual(leaf_index_map["doc_b"], [1])

    def test_build_navigation_samples_from_qa_maps_doc_ids_to_leaf_indices(self) -> None:
        tree_payload = build_tree_payload_from_corpus(
            [
                {"doc_id": "doc_a", "title": "Doc A", "text": "alpha beta gamma"},
                {"doc_id": "doc_b", "title": "Doc B", "text": "delta epsilon zeta"},
            ],
            max_chars_per_leaf=200,
        )

        batch_payload = build_navigation_samples_from_qa(
            [
                {
                    "sample_id": "sample_a",
                    "question": "What is in doc A?",
                    "reference_answer": "alpha beta gamma",
                    "positive_doc_ids": ["doc_a"],
                },
                {
                    "sample_id": "sample_ab",
                    "question": "What is in doc A and doc B?",
                    "reference_answer": "alpha beta gamma and delta epsilon zeta",
                    "positive_doc_ids": ["doc_a", "doc_b"],
                },
            ],
            tree_payload=tree_payload,
            tree_path="data/processed/real_corpus_tree_payload.json",
        )

        self.assertEqual(batch_payload["samples"][0]["positive_leaf_indices"], [0])
        self.assertEqual(batch_payload["samples"][1]["positive_leaf_indices"], [0, 1])
        self.assertEqual(
            batch_payload["samples"][1]["tree_path"],
            "data/processed/real_corpus_tree_payload.json",
        )


if __name__ == "__main__":
    unittest.main()
