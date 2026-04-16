from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.navigator import NavigatorState
from src.router import LearnedClassifierRouter, extract_router_features
from src.tree_builder import TreeNode


class LearnedRouterTest(unittest.TestCase):
    def test_extract_router_features_contains_expected_keys(self) -> None:
        features = extract_router_features(
            "What does relativity explain?",
            TreeNode(node_id="leaf", text="Relativity explains spacetime and gravity."),
            NavigatorState(relevance_score=0.5),
        )
        self.assertIn("lexical_overlap", features)
        self.assertIn("cosine_probe", features)
        self.assertIn("text_length_tokens", features)
        self.assertIn("parent_relevance", features)
        self.assertIn("child_is_leaf", features)

    def test_learned_classifier_router_prefers_weighted_child(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp_dir, True)
        checkpoint_path = temp_dir / "router.json"
        checkpoint = {
            "feature_names": ["lexical_overlap", "cosine_probe", "text_length_tokens", "parent_relevance", "child_is_leaf"],
            "weights": [1.0, 0.5, 0.0, 0.0, 0.0],
            "bias": 0.0,
        }
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

        router = LearnedClassifierRouter(checkpoint_path)
        child_good = TreeNode(node_id="good", text="Relativity explains gravity")
        child_bad = TreeNode(node_id="bad", text="apple orange banana")
        decision = router.rank_children(
            "What does relativity explain?",
            TreeNode(node_id="root", text="physics"),
            [child_bad, child_good],
            NavigatorState(),
        )
        self.assertEqual(decision.ordered_children[0].node_id, "good")

    def test_learned_classifier_rejects_unknown_checkpoint_features(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, temp_dir, True)
        checkpoint_path = temp_dir / "bad_router.json"
        checkpoint_path.write_text(
            json.dumps(
                {
                    "feature_names": ["lexical_overlap", "made_up_feature"],
                    "weights": [1.0, 1.0],
                    "bias": 0.0,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as ctx:
            LearnedClassifierRouter(checkpoint_path)
        self.assertIn("unknown feature", str(ctx.exception).lower())

    def test_extract_router_features_cached_question_matches_uncached(self) -> None:
        state = NavigatorState(relevance_score=0.25)
        node = TreeNode(node_id="n", text="Einstein and relativity.")
        from src.router.base import _text_vector, _tokenize_text

        question = "What is relativity about Einstein?"
        q_terms = set(_tokenize_text(question))
        q_vec = _text_vector(question)
        full = extract_router_features(question, node, state)
        cached = extract_router_features(
            question,
            node,
            state,
            question_terms=q_terms,
            question_vector=q_vec,
        )
        self.assertEqual(full, cached)


if __name__ == "__main__":
    unittest.main()
