#!/usr/bin/env python3
"""Export per-sample rows from accept_gate_audit report JSON into one CSV.

Adds a ``bucket`` column: ``never_visit`` | ``visit_miss`` | ``visit_clean`` | ``no_gold``.

Usage:

  python scripts/diagnostics/export_audit_sample_buckets.py \\
    --audit-json outputs/reports/accept_gate_audit_....json \\
    --out-csv outputs/reports/audit_buckets_export.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _bucket(row: dict[str, Any]) -> str:
    n_gold = int(row.get("n_gold_leaves") or 0)
    if n_gold <= 0:
        return "no_gold"
    hit = bool(row.get("gold_hit_visited"))
    n_na = int(row.get("n_gold_visited_not_accepted_leaves") or 0)
    if not hit:
        return "never_visit"
    if n_na > 0:
        return "visit_miss"
    if row.get("all_visited_gold_in_accepted") is True:
        return "visit_clean"
    return "visit_other"


def _disp_str(row: dict[str, Any]) -> str:
    d = row.get("visited_not_accepted_dispositions") or {}
    if not isinstance(d, dict) or not d:
        return ""
    parts = [f"{k}={v}" for k, v in sorted(d.items()) if isinstance(v, int)]
    return ";".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-json", type=str, required=True)
    p.add_argument(
        "--out-csv",
        type=str,
        default="outputs/reports/audit_sample_buckets_export.csv",
        help="Output CSV path (relative to cwd unless absolute)",
    )
    ns = p.parse_args(argv)

    path = Path(ns.audit_json)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 2

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows_in = data.get("per_sample")
    if not isinstance(rows_in, list):
        print("Missing per_sample", file=sys.stderr)
        return 2

    bid = None
    inp = data.get("inputs")
    if isinstance(inp, dict):
        bid = inp.get("batch_id")

    out_path = Path(ns.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "batch_id",
        "bucket",
        "sample_id",
        "run_id",
        "n_gold_leaves",
        "n_gold_never_visited",
        "n_gold_visited_not_accepted_leaves",
        "n_gold_leaves_accepted",
        "gold_hit_visited",
        "all_visited_gold_in_accepted",
        "visited_leaf_deduped_count",
        "n_evidence",
        "visit_miss_dispositions",
        "source_path",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows_in:
            if not isinstance(row, dict):
                continue
            w.writerow(
                {
                    "batch_id": bid or "",
                    "bucket": _bucket(row),
                    "sample_id": row.get("sample_id", ""),
                    "run_id": row.get("run_id", ""),
                    "n_gold_leaves": int(row.get("n_gold_leaves") or 0),
                    "n_gold_never_visited": int(row.get("n_gold_never_visited") or 0),
                    "n_gold_visited_not_accepted_leaves": int(
                        row.get("n_gold_visited_not_accepted_leaves") or 0
                    ),
                    "n_gold_leaves_accepted": int(row.get("n_gold_leaves_accepted") or 0),
                    "gold_hit_visited": row.get("gold_hit_visited", ""),
                    "all_visited_gold_in_accepted": row.get("all_visited_gold_in_accepted", ""),
                    "visited_leaf_deduped_count": int(row.get("visited_leaf_deduped_count") or 0),
                    "n_evidence": int(row.get("n_evidence") or 0),
                    "visit_miss_dispositions": _disp_str(row),
                    "source_path": row.get("source_path", ""),
                }
            )

    print(f"Wrote {len(rows_in)} rows to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
