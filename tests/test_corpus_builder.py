from __future__ import annotations

import unittest

from src.tree_builder import build_tree_payload_from_corpus


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


if __name__ == "__main__":
    unittest.main()
