from __future__ import annotations

import unittest

from src.routing.entity_match import (
    apply_entity_boost,
    compute_entity_hit_rate,
    compute_entity_match_score,
    entity_mentioned_in_text,
    extract_question_entities,
    keyword_token_overlap_fraction,
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

    def test_entity_word_boundary_avoids_substring_false_positive(self) -> None:
        self.assertFalse(entity_mentioned_in_text("apple", "pineapple pie".lower()))
        self.assertTrue(entity_mentioned_in_text("apple", "an apple pie".lower()))

    def test_extract_question_entities_includes_acronyms(self) -> None:
        entities = extract_question_entities("Did the NBA or NFL win more titles in the 1990s?")
        lowered = [e.lower() for e in entities]
        self.assertIn("nba", lowered)
        self.assertIn("nfl", lowered)

    def test_filter_sentence_lead_drops_isolated_lead_function_word(self) -> None:
        entities = extract_question_entities(
            "Here we only discuss Newton and Einstein.",
            filter_sentence_lead=True,
        )
        self.assertNotIn("Here", entities)
        self.assertTrue(any("Newton" in e for e in entities))

    def test_filter_sentence_lead_can_be_disabled(self) -> None:
        entities = extract_question_entities(
            "Here we only discuss Newton and Einstein.",
            filter_sentence_lead=False,
        )
        self.assertIn("Here", entities)

    def test_keyword_token_overlap_fraction_basic(self) -> None:
        q = "What is the capital of France?"
        text = "Paris is the capital city of France."
        self.assertEqual(keyword_token_overlap_fraction(q, text), 1.0)

    def test_keyword_token_overlap_fraction_partial(self) -> None:
        q = "Did Newton or Einstein propose relativity?"
        text = "Einstein proposed special relativity in 1905."
        ov = keyword_token_overlap_fraction(q, text)
        self.assertGreater(ov, 0.0)
        self.assertLess(ov, 1.0)

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
        self.assertEqual(boosted_sorted[0]["raw_router_score"], 0.1)
        self.assertEqual(boosted_sorted[1]["raw_router_score"], 0.1)

        hit_rate, intersection_size = compute_entity_hit_rate(
            question_entities=["Einstein", "Relativity"],
            visited_leaf_texts=["Einstein proposed relativity.", "Newton studied calculus."],
        )
        self.assertEqual(intersection_size, 2)
        self.assertEqual(hit_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
