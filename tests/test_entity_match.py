from __future__ import annotations

import unittest

from src.routing.entity_match import (
    apply_entity_boost,
    compute_entity_hit_rate,
    compute_entity_match_score,
    extract_question_entities,
)


class EntityMatchTest(unittest.TestCase):
    def test_extract_question_entities_prefers_capitalized_spans(self) -> None:
        entities = extract_question_entities(
            "Which film was released more recently, Royal Treasure or When Love Begins?"
        )
        self.assertIn("Royal Treasure", entities)
        self.assertIn("When Love Begins", entities)

    def test_compute_entity_match_score_is_fractional(self) -> None:
        score = compute_entity_match_score(
            question_entities=["Royal Treasure", "When Love Begins"],
            node_text="Royal Treasure was released in 2016.",
        )
        self.assertEqual(score, 0.5)

    def test_apply_entity_boost_and_hit_rate(self) -> None:
        scored_children = [
            {"node_id": "leaf_a", "score": 0.1},
            {"node_id": "leaf_b", "score": 0.1},
        ]
        node_text_map = {
            "leaf_a": "Newton discussed gravity.",
            "leaf_b": "Einstein proposed relativity.",
        }
        boosted = apply_entity_boost(
            scored_children=scored_children,
            question_entities=["Einstein"],
            alpha=0.5,
            get_node_text=lambda node_id: node_text_map[node_id],
        )
        boosted_sorted = sorted(boosted, key=lambda row: row["score"], reverse=True)
        self.assertEqual(boosted_sorted[0]["node_id"], "leaf_b")
        self.assertEqual(boosted_sorted[0]["entity_match_score"], 1.0)

        hit_rate, intersection_size = compute_entity_hit_rate(
            question_entities=["Einstein", "Relativity"],
            visited_leaf_texts=["Einstein proposed relativity.", "Newton studied calculus."],
        )
        self.assertEqual(intersection_size, 2)
        self.assertEqual(hit_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
