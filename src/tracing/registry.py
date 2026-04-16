from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .report_fields import trace_fields_for_reports


def build_registry_row(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace", {})
    config = payload.get("config", {})
    common = trace_fields_for_reports(trace, config=config)

    return {
        "run_id": payload.get("run_id"),
        "sample_id": payload.get("sample_id"),
        "batch_id": payload.get("batch_id"),
        "eval_mode": payload.get("eval_mode"),
        "question": payload.get("question"),
        "tree_path": payload.get("tree_path"),
        "navigator_type": config.get("navigator_type", "mock"),
        "navigator_model_name": config.get("navigator_model_name"),
        "generator_type": config.get("generator_type", "qwen"),
        "generator_model_name": config.get("generator_model_name"),
        "context_source": trace.get("context_source", config.get("context_source")),
        **common,
        "output_run_dir": payload.get("output_run_dir"),
    }


def append_jsonl(path: str | Path, row: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path
