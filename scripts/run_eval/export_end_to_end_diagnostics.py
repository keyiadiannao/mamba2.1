from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export end-to-end diagnostic samples across batches.")
    parser.add_argument(
        "--batch-id",
        action="append",
        required=True,
        help="Exact batch_id to include. Repeat this flag for multiple batches.",
    )
    parser.add_argument(
        "--priority-batch-id",
        default=None,
        help="Batch used to sort samples by ascending answer_f1 for diagnosis.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Maximum number of aligned samples to export.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/end_to_end_diagnostics.jsonl",
        help="Path to the exported diagnostic jsonl.",
    )
    parser.add_argument(
        "--registry-input",
        default="outputs/reports/run_registry.jsonl",
        help="Path to the run registry jsonl.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def load_run_payload(run_dir: str | Path) -> dict[str, Any]:
    payload_path = Path(run_dir) / "run_payload.json"
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_case_entry(batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace", {})
    config = payload.get("config", {})
    return {
        "batch_id": batch_id,
        "run_id": payload.get("run_id"),
        "routing_mode": trace.get("routing_mode", config.get("routing_mode")),
        "context_source": trace.get("context_source", config.get("context_source")),
        "nav_success": trace.get("nav_success"),
        "exact_match": trace.get("exact_match"),
        "answer_f1": trace.get("answer_f1"),
        "rouge_l_f1": trace.get("rouge_l_f1"),
        "rollback_count": trace.get("rollback_count"),
        "nav_wall_time_ms": trace.get("nav_wall_time_ms"),
        "generation_error": trace.get("generation_error"),
        "context_build_error": trace.get("context_build_error"),
        "generator_prompt": payload.get("generator_prompt"),
        "generator_evidence_texts": payload.get("generator_evidence_texts", []),
        "generator_evidence_node_ids": payload.get("generator_evidence_node_ids", []),
        "generated_answer": payload.get("generated_answer"),
        "output_run_dir": payload.get("output_run_dir"),
    }


def main() -> None:
    args = parse_args()
    batch_ids = list(dict.fromkeys(args.batch_id))
    priority_batch_id = args.priority_batch_id or batch_ids[0]
    if priority_batch_id not in batch_ids:
        raise ValueError("priority_batch_id must also be included in --batch-id.")

    registry_rows = load_jsonl(ROOT / Path(args.registry_input))
    selected_rows = [
        row
        for row in registry_rows
        if row.get("batch_id") in batch_ids and row.get("sample_id") and row.get("output_run_dir")
    ]
    if not selected_rows:
        raise ValueError("No matching rows found for the provided batch ids.")

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in selected_rows:
        sample_id = str(row["sample_id"])
        batch_id = str(row["batch_id"])
        grouped.setdefault(sample_id, {})[batch_id] = row

    aligned_sample_ids = [
        sample_id
        for sample_id, sample_rows in grouped.items()
        if all(batch_id in sample_rows for batch_id in batch_ids)
    ]
    if not aligned_sample_ids:
        raise ValueError("No sample_ids were found in all requested batches.")

    def sort_key(sample_id: str) -> tuple[float, str]:
        priority_row = grouped[sample_id][priority_batch_id]
        answer_f1 = priority_row.get("answer_f1")
        if answer_f1 is None:
            return (-1.0, sample_id)
        return (float(answer_f1), sample_id)

    aligned_sample_ids = sorted(aligned_sample_ids, key=sort_key)
    limited_sample_ids = aligned_sample_ids[: max(args.sample_limit, 0)]

    diagnostic_rows: list[dict[str, Any]] = []
    for sample_id in limited_sample_ids:
        sample_rows = grouped[sample_id]
        priority_payload = load_run_payload(sample_rows[priority_batch_id]["output_run_dir"])
        cases = []
        for batch_id in batch_ids:
            payload = load_run_payload(sample_rows[batch_id]["output_run_dir"])
            cases.append(build_case_entry(batch_id, payload))

        diagnostic_rows.append(
            {
                "sample_id": sample_id,
                "question": priority_payload.get("question"),
                "reference_answer": priority_payload.get("reference_answer"),
                "cases": cases,
            }
        )

    output_path = ROOT / Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in diagnostic_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "batch_ids": batch_ids,
                "priority_batch_id": priority_batch_id,
                "aligned_sample_count": len(aligned_sample_ids),
                "exported_sample_count": len(diagnostic_rows),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
