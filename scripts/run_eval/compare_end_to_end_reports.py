from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare end-to-end evaluation summaries.")
    parser.add_argument(
        "--input",
        default="outputs/reports/navigation_summary.jsonl",
        help="Path to the navigation summary jsonl file.",
    )
    parser.add_argument(
        "--navigator-type",
        default=None,
        help="Optional navigator_type filter, e.g. mamba_ssm.",
    )
    parser.add_argument(
        "--batch-id",
        action="append",
        default=None,
        help="Optional exact batch_id filter; may be provided multiple times.",
    )
    parser.add_argument(
        "--batch-id-contains",
        default=None,
        help="Optional substring filter on batch_id.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input report not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(ROOT / Path(args.input))
    filtered: list[dict[str, object]] = []
    batch_ids = set(args.batch_id or [])
    for row in rows:
        if args.navigator_type and row.get("navigator_type") != args.navigator_type:
            continue
        if batch_ids and row.get("batch_id") not in batch_ids:
            continue
        if args.batch_id_contains and args.batch_id_contains not in str(row.get("batch_id")):
            continue
        filtered.append(row)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in filtered:
        context_source = str(row.get("context_source") or "unknown_context")
        routing_mode = str(row.get("routing_mode") or "unknown_routing")
        group_key = f"{context_source}__{routing_mode}"
        grouped[group_key].append(row)

    comparison: dict[str, dict[str, object]] = {}
    for group_key, group_rows in grouped.items():
        sample_count = len(group_rows)
        nav_successes = sum(1 for row in group_rows if row.get("nav_success"))
        exact_match_hits = sum(1 for row in group_rows if row.get("exact_match") == 1)
        answer_f1_values = [float(row["answer_f1"]) for row in group_rows if row.get("answer_f1") is not None]
        rouge_values = [float(row["rouge_l_f1"]) for row in group_rows if row.get("rouge_l_f1") is not None]
        nav_time_values = [float(row.get("nav_wall_time_ms") or 0.0) for row in group_rows]
        rollback_values = [int(row.get("rollback_count") or 0) for row in group_rows]
        evidence_values = [int(row.get("evidence_count") or 0) for row in group_rows]
        context_values = [int(row.get("context_item_count") or 0) for row in group_rows]

        comparison[group_key] = {
            "sample_count": sample_count,
            "context_source": group_rows[0].get("context_source"),
            "routing_mode": group_rows[0].get("routing_mode"),
            "nav_success_rate": (nav_successes / sample_count) if sample_count else 0.0,
            "exact_match_rate": (exact_match_hits / sample_count) if sample_count else 0.0,
            "avg_answer_f1": (sum(answer_f1_values) / len(answer_f1_values)) if answer_f1_values else 0.0,
            "avg_rouge_l_f1": (sum(rouge_values) / len(rouge_values)) if rouge_values else 0.0,
            "avg_nav_wall_time_ms": (sum(nav_time_values) / sample_count) if sample_count else 0.0,
            "avg_rollback_count": (sum(rollback_values) / sample_count) if sample_count else 0.0,
            "avg_evidence_count": (sum(evidence_values) / sample_count) if sample_count else 0.0,
            "avg_context_item_count": (sum(context_values) / sample_count) if sample_count else 0.0,
            "sample_ids": [row.get("sample_id") for row in group_rows],
        }

    output = {
        "input_path": str(ROOT / Path(args.input)),
        "navigator_type_filter": args.navigator_type,
        "batch_ids_filter": list(batch_ids) if batch_ids else None,
        "batch_id_contains_filter": args.batch_id_contains,
        "row_count": len(filtered),
        "comparison": comparison,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
