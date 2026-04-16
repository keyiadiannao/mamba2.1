from __future__ import annotations

from typing import Any


def trace_fields_for_reports(
    trace: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fields derived from ``trace`` shared by registry and navigation summary.

    Keeps one definition for duplicated scalars to reduce drift when new metrics
    are added to the trace schema.
    """
    cfg = config or {}
    evidence_texts = trace.get("evidence_texts") or []
    return {
        "routing_mode": trace.get("routing_mode", cfg.get("routing_mode")),
        "nav_success": trace.get("nav_success"),
        "rollback_count": trace.get("rollback_count"),
        "snapshot_stack_max_depth": trace.get("snapshot_stack_max_depth"),
        "snapshot_push_count": trace.get("snapshot_push_count"),
        "snapshot_restore_count": trace.get("snapshot_restore_count"),
        "nav_wall_time_ms": trace.get("nav_wall_time_ms"),
        "visited_leaf_count": len(trace.get("visited_leaf_indices_deduped") or []),
        "evidence_count": len(evidence_texts),
        "context_item_count": len(trace.get("context_texts") or []),
        "exact_match": trace.get("exact_match"),
        "answer_f1": trace.get("answer_f1"),
        "rouge_l_f1": trace.get("rouge_l_f1"),
        "postprocess_mode": trace.get("postprocess_mode"),
        "postprocess_rule": trace.get("postprocess_rule"),
        "entity_boost_alpha": trace.get("entity_boost_alpha"),
        "question_entity_count": trace.get("question_entity_count"),
        "entity_intersection_size": trace.get("entity_intersection_size"),
        "entity_hit_rate": trace.get("entity_hit_rate"),
        "raw_generated_answer": trace.get("raw_generated_answer"),
        "postprocessed_answer": trace.get("postprocessed_answer"),
        "generation_error": trace.get("generation_error"),
    }
