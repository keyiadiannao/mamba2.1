from __future__ import annotations

"""Audit router-training jsonl for label consistency and missing positive rows.

Typical usage (repo root):

  py -3 scripts/run_nav/audit_router_training_jsonl.py \\
    --input outputs/reports/router_training_data_root_v2.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        help="Path to router training jsonl (relative to repo root or absolute).",
    )
    p.add_argument(
        "--max-report",
        type=int,
        default=20,
        help="Max sample_ids to print per issue category (default 20).",
    )
    return p.parse_args()


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path} line {line_no}: {exc}") from exc
    return rows


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    if not path.is_absolute():
        path = ROOT / path
    rows = _load_rows(path)

    row_errors: list[str] = []
    by_sample: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        sid = str(row.get("sample_id", ""))
        by_sample[sid].append(row)

    gold_nonempty_samples = 0
    samples_missing_positive_row: list[str] = []
    samples_multiple_positive_children: list[str] = []

    for sid, sample_rows in sorted(by_sample.items(), key=lambda x: x[0]):
        gold = {int(x) for x in (sample_rows[0].get("positive_leaf_indices") or []) if isinstance(x, int)}
        if gold:
            gold_nonempty_samples += 1

        pos_children: set[str] = set()
        for row in sample_rows:
            child_leaves = {
                int(x) for x in (row.get("child_leaf_indices") or []) if isinstance(x, int)
            }
            gold_set = {
                int(x) for x in (row.get("positive_leaf_indices") or []) if isinstance(x, int)
            }
            hit = bool(child_leaves.intersection(gold_set))
            label = int(row.get("label", -1))
            if hit != (label == 1):
                row_errors.append(
                    f"sample_id={sid} child={row.get('child_node_id')}: "
                    f"label={label} but gold∩child_leaves={'nonempty' if hit else 'empty'}"
                )
            if label == 1:
                pos_children.add(str(row.get("child_node_id", "")))

        if gold and not pos_children:
            samples_missing_positive_row.append(sid)
        if len(pos_children) > 1:
            samples_multiple_positive_children.append(sid)

    report: dict[str, object] = {
        "input_path": str(path),
        "row_count": len(rows),
        "sample_count": len(by_sample),
        "samples_with_nonempty_gold": gold_nonempty_samples,
        "row_label_inconsistent_with_leaf_sets": len(row_errors),
        "samples_with_gold_but_no_positive_row_in_export": len(samples_missing_positive_row),
        "samples_with_multiple_distinct_positive_root_children": len(samples_multiple_positive_children),
    }

    mr = max(0, int(args.max_report))
    if row_errors:
        report["row_errors_sample"] = row_errors[:mr]
    if samples_missing_positive_row:
        report["missing_positive_sample_ids_sample"] = samples_missing_positive_row[:mr]
    if samples_multiple_positive_children:
        report["multi_positive_root_children_sample_ids_sample"] = samples_multiple_positive_children[:mr]

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if row_errors or samples_missing_positive_row:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
