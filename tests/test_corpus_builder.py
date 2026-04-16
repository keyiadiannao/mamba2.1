from __future__ import annotations

import unittest

from src.tree_builder import build_tree_payload_from_corpus
from src.tree_builder.corpus_builder import _chunk_by_sentences


class CorpusBuilderTest(unittest.TestCase):
    def test_build_tree_payload_from_corpus_creates_branch_and_leaf_nodes(self) -> None:
        payload = build_tree_payload_from_corpus(
            [
                {
                    "doc_id": "einstein_notes",
                    "title": "Einstein Notes",
                    "summary": "Relativity overview",
                    "text": "Einstein proposed special relativity and general relativity.",
                }
            ],
            question="What did Einstein propose?",
            reference_answer="Einstein proposed special relativity and general relativity.",
            max_chars_per_leaf=120,
        )

        self.assertEqual(payload["question"], "What did Einstein propose?")
        self.assertEqual(payload["reference_answer"], "Einstein proposed special relativity and general relativity.")
        self.assertEqual(payload["root"]["children"][0]["node_id"], "branch_einstein_notes")
        self.assertEqual(payload["root"]["children"][0]["children"][0]["leaf_index"], 0)
        self.assertIsNotNone(payload.get("tree_sha256"))
        self.assertEqual(len(str(payload["tree_sha256"])), 64)
        root_meta = payload["root"]["metadata"]
        self.assertIsNone(root_meta.get("parent_id"))
        self.assertEqual(root_meta.get("depth"), 0)
        self.assertEqual(root_meta.get("path_node_ids"), ["root"])
        leaf_meta = payload["root"]["children"][0]["children"][0]["metadata"]
        self.assertEqual(leaf_meta.get("parent_id"), "branch_einstein_notes")
        self.assertEqual(leaf_meta.get("depth"), 2)

    def test_build_tree_payload_from_corpus_chunks_long_documents(self) -> None:
        long_text = " ".join(["relativity"] * 80)
        payload = build_tree_payload_from_corpus(
            [{"doc_id": "long_doc", "title": "Long Doc", "text": long_text}],
            max_chars_per_leaf=80,
        )

        leaf_nodes = payload["root"]["children"][0]["children"]
        self.assertGreater(len(leaf_nodes), 1)
        self.assertEqual(leaf_nodes[0]["metadata"]["chunk_index"], 0)
        self.assertEqual(leaf_nodes[-1]["metadata"]["chunk_index"], len(leaf_nodes) - 1)

    def test_build_tree_payload_from_corpus_splits_on_sentence_boundaries(self) -> None:
        payload = build_tree_payload_from_corpus(
            [
                {
                    "doc_id": "two_sent",
                    "title": "Two Sent",
                    "text": "First sentence here. Second sentence follows.",
                }
            ],
            max_chars_per_leaf=22,
        )
        leaves = payload["root"]["children"][0]["children"]
        self.assertGreaterEqual(len(leaves), 2)
        self.assertTrue(any("First" in leaf["text"] for leaf in leaves))
        self.assertTrue(any("Second" in leaf["text"] for leaf in leaves))

    def test_tree_sha256_is_stable_for_identical_tree(self) -> None:
        records = [{"doc_id": "a", "title": "A", "text": "Hello world."}]
        p1 = build_tree_payload_from_corpus(records, max_chars_per_leaf=200)
        p2 = build_tree_payload_from_corpus(records, max_chars_per_leaf=200)
        self.assertEqual(p1["tree_sha256"], p2["tree_sha256"])

    def test_chunk_by_sentences_rejects_nonzero_overlap(self) -> None:
        with self.assertRaises(ValueError):
            _chunk_by_sentences("A. B.", 10, overlap=1)


if __name__ == "__main__":
    unittest.main()
