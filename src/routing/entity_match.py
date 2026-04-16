"""Entity-aware routing boost: extract candidate entities from the question
and compute a match score against each child node's text.

This module is intentionally lightweight — no NER model, just capitalized-span
extraction — so the signal is a fair, non-cheating probe for whether the
navigator is walking towards the right entities.
"""

from __future__ import annotations

import re
from typing import Any


def extract_question_entities(question: str) -> list[str]:
    """Extract candidate entity spans from a question string.

    Strategy (ordered by priority):
    1. Quoted strings — "Film X" or 'Film X'
    2. Capitalized spans of length >= 2 — "Royal Treasure", "New York"
    3. Fallback: unique non-stopword tokens of length >= 4

    Returns a deduplicated list of entity strings (original casing preserved).
    """
    entities: list[str] = []

    # 1) Quoted strings
    quoted = re.findall(r'["\']([^"\']+)["\']', question)
    entities.extend(quoted)

    _STOPWORDS = {
        "What", "Which", "Who", "Whom", "Whose", "Where", "When", "How",
        "Why", "Does", "Do", "Did", "Is", "Are", "Was", "Were",
        "The", "A", "An", "And", "Or", "But", "In", "On", "At", "To",
        "For", "Of", "With", "By", "From", "Into", "During",
        "This", "That", "These", "Those", "Not", "No",
    }
    # 2) Capitalized spans (at least 2 chars, starting with uppercase)
    cap_spans = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", question)
    entities.extend(span for span in cap_spans if span not in _STOPWORDS)

    # 3) Fallback: long non-stopword tokens
    if not entities:
        tokens = re.findall(r"\b([A-Za-z]{4,})\b", question)
        entities.extend(t for t in tokens if t not in _STOPWORDS)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for ent in entities:
        key = ent.lower()
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    return unique


def compute_entity_match_score(
    question_entities: list[str],
    node_text: str,
) -> float:
    """Fraction of question entities that appear (case-insensitive) in node_text.

    Returns a value in [0.0, 1.0].  Returns 0.0 if there are no question
    entities (avoids division by zero and also means "no signal" rather than
    a spurious 1.0).
    """
    if not question_entities:
        return 0.0
    node_lower = node_text.lower()
    hits = sum(1 for ent in question_entities if ent.lower() in node_lower)
    return hits / len(question_entities)


def apply_entity_boost(
    scored_children: list[dict[str, Any]],
    question_entities: list[str],
    alpha: float,
    get_node_text: callable,
) -> list[dict[str, Any]]:
    """Add entity_match_score * alpha to each child's routing score.

    Args:
        scored_children: List of dicts with at least "score" and "node_id".
        question_entities: Entities extracted from the question.
        alpha: Boost weight. 0.0 = no boost (baseline), 0.3 = conservative,
               0.5 = aggressive.
        get_node_text: Callable(node_id) -> str, returns node text for matching.

    Returns:
        New list with updated "score" and added "entity_match_score" per child.
    """
    if alpha <= 0.0 or not question_entities:
        return scored_children

    boosted: list[dict[str, Any]] = []
    for child in scored_children:
        child = dict(child)
        node_text = get_node_text(child.get("node_id", ""))
        ems = compute_entity_match_score(question_entities, node_text)
        child["entity_match_score"] = ems
        child["score"] = float(child.get("score", 0.0)) + alpha * ems
        boosted.append(child)
    return boosted


def compute_entity_hit_rate(
    question_entities: list[str],
    visited_leaf_texts: list[str],
) -> tuple[float, int]:
    """Compute what fraction of question entities appear in visited leaves.

    Args:
        question_entities: Entities extracted from the question.
        visited_leaf_texts: Texts of all visited leaf nodes.

    Returns:
        (hit_rate, intersection_size) — hit_rate in [0.0, 1.0],
        intersection_size is the count of matched entities.
        Returns (0.0, 0) if there are no question entities.
    """
    if not question_entities:
        return 0.0, 0

    all_text_lower = " ".join(visited_leaf_texts).lower()
    intersection_size = sum(
        1 for ent in question_entities if ent.lower() in all_text_lower
    )
    hit_rate = intersection_size / len(question_entities)
    return hit_rate, intersection_size
