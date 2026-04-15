from __future__ import annotations

import unittest

from src.tree_builder import (
    build_corpus_and_qa_from_wiki_longdoc_samples,
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

    def test_build_corpus_and_qa_from_wiki_longdoc_samples_expands_pages_and_sections(self) -> None:
        corpus_records, qa_records = build_corpus_and_qa_from_wiki_longdoc_samples(
            [
                {
                    "sample_id": "wiki_q1",
                    "question": "Where did the scientist work?",
                    "reference_answer": "At the Cavendish Laboratory.",
                    "supporting_page_ids": ["page_scientist"],
                    "pages": [
                        {
                            "page_id": "page_scientist",
                            "title": "Scientist",
                            "lead_text": "Scientist overview",
                            "sections": [
                                {
                                    "section_id": "page_scientist__career",
                                    "heading": "Career",
                                    "paragraphs": [
                                        "The scientist worked at the Cavendish Laboratory.",
                                        "The scientist later taught at the university.",
                                    ],
                                },
                                {
                                    "section_id": "page_scientist__awards",
                                    "heading": "Awards",
                                    "paragraphs": ["The scientist won a major prize."],
                                },
                            ],
                        }
                    ],
                }
            ],
            source_name="wiki_subset_test",
        )

        self.assertEqual(len(corpus_records), 2)
        self.assertEqual(corpus_records[0]["group_id"], "page_scientist")
        self.assertEqual(corpus_records[0]["source"], "wiki_subset_test")
        self.assertEqual(qa_records[0]["sample_id"], "wiki_q1")
        self.assertEqual(
            qa_records[0]["positive_doc_ids"],
            ["page_scientist__awards", "page_scientist__career"],
        )

    def test_grouped_tree_payload_and_leaf_mapping_work_for_wiki_longdoc_records(self) -> None:
        corpus_records, qa_records = build_corpus_and_qa_from_wiki_longdoc_samples(
            [
                {
                    "sample_id": "wiki_q2",
                    "question": "What did the city build?",
                    "reference_answer": "A stone bridge.",
                    "supporting_section_ids": ["page_city__history"],
                    "pages": [
                        {
                            "page_id": "page_city",
                            "title": "City",
                            "lead_text": "City overview",
                            "sections": [
                                {
                                    "section_id": "page_city__history",
                                    "heading": "History",
                                    "paragraphs": [
                                        "The city built a stone bridge.",
                                        "The bridge connected two districts.",
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        )

        tree_payload = build_tree_payload_from_corpus(corpus_records, max_chars_per_leaf=500)
        root_children = tree_payload["root"]["children"]
        self.assertEqual(len(root_children), 1)
        self.assertEqual(root_children[0]["node_id"], "group_page_city")
        self.assertEqual(root_children[0]["children"][0]["node_id"], "branch_page_city__history")

        batch_payload = build_navigation_samples_from_qa(
            qa_records,
            tree_payload=tree_payload,
            tree_path="data/processed/wiki_longdoc_tree_payload.json",
        )
        self.assertEqual(batch_payload["samples"][0]["positive_leaf_indices"], [0])
        self.assertEqual(
            batch_payload["samples"][0]["tree_path"],
            "data/processed/wiki_longdoc_tree_payload.json",
        )


if __name__ == "__main__":
    unittest.main()
