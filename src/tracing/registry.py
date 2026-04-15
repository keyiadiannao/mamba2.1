from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_registry_row(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace", {})
    config = payload.get("config", {})
    evidence_texts = trace.get("evidence_texts", [])

    return {
        "run_id": payload.get("run_id"),
        "sample_id": payload.get("sample_id"),
        "batch_id": payload.get("batch_id"),
        "question": payload.get("question"),
        "tree_path": payload.get("tree_path"),
        "navigator_type": config.get("navigator_type", "mock"),
        "navigator_model_name": config.get("navigator_model_name"),
        "generator_type": config.get("generator_type", "qwen"),
        "generator_model_name": config.get("generator_model_name"),
        "routing_mode": trace.get("routing_mode", config.get("routing_mode")),
        "context_source": trace.get("context_source", config.get("context_source")),
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
        "generation_error": trace.get("generation_error"),
        "output_run_dir": payload.get("output_run_dir"),
    }


def append_jsonl(path: str | Path, row: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path
