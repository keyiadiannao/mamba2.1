from __future__ import annotations

from typing import Any


def build_navigation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace", {})
    config = payload.get("config", {})

    return {
        "run_id": payload.get("run_id"),
        "question": payload.get("question"),
        "navigator_type": config.get("navigator_type", "mock"),
        "routing_mode": trace.get("routing_mode", config.get("routing_mode")),
        "nav_success": trace.get("nav_success"),
        "failure_attribution": trace.get("failure_attribution"),
        "visited_node_count": len(trace.get("visited_node_ids") or []),
        "visited_leaf_count": len(trace.get("visited_leaf_indices_deduped") or []),
        "evidence_count": len(trace.get("evidence_texts") or []),
        "rollback_count": trace.get("rollback_count"),
        "snapshot_stack_max_depth": trace.get("snapshot_stack_max_depth"),
        "nav_wall_time_ms": trace.get("nav_wall_time_ms"),
        "context_build_error": trace.get("context_build_error"),
        "evidence_node_ids": trace.get("evidence_node_ids") or [],
    }
