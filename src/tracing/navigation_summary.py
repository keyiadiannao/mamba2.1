from __future__ import annotations

from typing import Any


def build_navigation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace", {})
    config = payload.get("config", {})

    return {
        "run_id": payload.get("run_id"),
        "sample_id": payload.get("sample_id"),
        "batch_id": payload.get("batch_id"),
        "question": payload.get("question"),
        "navigator_type": config.get("navigator_type", "mock"),
        "routing_mode": trace.get("routing_mode", config.get("routing_mode")),
        "nav_success": trace.get("nav_success"),
        "failure_attribution": trace.get("failure_attribution"),
        "visited_node_count": len(trace.get("visited_node_ids") or []),
        "visited_leaf_count": len(trace.get("visited_leaf_indices_deduped") or []),
        "evidence_count": len(trace.get("evidence_texts") or []),
        "context_item_count": len(trace.get("context_texts") or []),
        "rollback_count": trace.get("rollback_count"),
        "snapshot_stack_max_depth": trace.get("snapshot_stack_max_depth"),
        "snapshot_push_count": trace.get("snapshot_push_count"),
        "snapshot_restore_count": trace.get("snapshot_restore_count"),
        "nav_wall_time_ms": trace.get("nav_wall_time_ms"),
        "context_build_error": trace.get("context_build_error"),
        "exact_match": trace.get("exact_match"),
        "answer_f1": trace.get("answer_f1"),
        "rouge_l_f1": trace.get("rouge_l_f1"),
        "generation_error": trace.get("generation_error"),
        "evidence_node_ids": trace.get("evidence_node_ids") or [],
        "context_node_ids": trace.get("context_node_ids") or [],
    }
