from __future__ import annotations

from typing import Any

from .report_fields import trace_fields_for_reports


def build_navigation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace", {})
    config = payload.get("config", {})
    common = trace_fields_for_reports(trace, config=config)

    return {
        "run_id": payload.get("run_id"),
        "sample_id": payload.get("sample_id"),
        "batch_id": payload.get("batch_id"),
        "question": payload.get("question"),
        "navigator_type": config.get("navigator_type", "mock"),
        **common,
        "failure_attribution": trace.get("failure_attribution"),
        "visited_node_count": len(trace.get("visited_node_ids") or []),
        "context_build_error": trace.get("context_build_error"),
        "evidence_node_ids": trace.get("evidence_node_ids") or [],
        "context_node_ids": trace.get("context_node_ids") or [],
    }
