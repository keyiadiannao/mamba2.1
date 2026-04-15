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
    parser = argparse.ArgumentParser(description="Compare navigation summaries by routing mode.")
    parser.add_argument(
        "--input",
        default="outputs/reports/navigation_summary.jsonl",
        help="Path to the navigation summary jsonl file.",
    )
    parser.add_argument(
        "--navigator-type",
        default=None,
        help="Optional navigator_type filter, e.g. mamba_ssm or mock.",
    )
    parser.add_argument(
        "--run-id-contains",
        default=None,
        help="Optional substring filter on run_id.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional exact batch_id filter.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"Input report not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(ROOT / Path(args.input))

    filtered = []
    for row in rows:
        if args.navigator_type and row.get("navigator_type") != args.navigator_type:
            continue
        if args.run_id_contains and args.run_id_contains not in str(row.get("run_id")):
            continue
        if args.batch_id and row.get("batch_id") != args.batch_id:
            continue
        filtered.append(row)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in filtered:
        grouped[str(row.get("routing_mode", "unknown"))].append(row)

    comparison = {}
    for routing_mode, mode_rows in grouped.items():
        sample_count = len(mode_rows)
        nav_successes = sum(1 for row in mode_rows if row.get("nav_success"))
        total_nav_time = sum(float(row.get("nav_wall_time_ms") or 0.0) for row in mode_rows)
        total_rollbacks = sum(int(row.get("rollback_count") or 0) for row in mode_rows)
        total_evidence = sum(int(row.get("evidence_count") or 0) for row in mode_rows)
        comparison[routing_mode] = {
            "sample_count": sample_count,
            "nav_success_rate": (nav_successes / sample_count) if sample_count else 0.0,
            "avg_nav_wall_time_ms": (total_nav_time / sample_count) if sample_count else 0.0,
            "avg_rollback_count": (total_rollbacks / sample_count) if sample_count else 0.0,
            "avg_evidence_count": (total_evidence / sample_count) if sample_count else 0.0,
            "sample_ids": [row.get("sample_id") for row in mode_rows],
        }

    output = {
        "input_path": str(ROOT / Path(args.input)),
        "navigator_type_filter": args.navigator_type,
        "run_id_contains_filter": args.run_id_contains,
        "batch_id_filter": args.batch_id,
        "row_count": len(filtered),
        "comparison": comparison,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
