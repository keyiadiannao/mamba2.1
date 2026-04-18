#!/usr/bin/env python3
"""Print compact tables from saved diagnostic JSON reports (terminal-friendly).

Supports full reports written by:
  - scripts/diagnostics/audit_accept_gate.py (--out-json)
  - scripts/diagnostics/analyze_evidence_saturation.py (--out-json)

Usage (from repo root):

  python scripts/diagnostics/print_diagnostic_summaries.py \\
    outputs/reports/accept_gate_audit_nav_....json \\
    outputs/reports/evidence_saturation_nav_....json

  python scripts/diagnostics/print_diagnostic_summaries.py --glob 'outputs/reports/accept_gate_audit_*.json'

  python scripts/diagnostics/print_diagnostic_summaries.py --tsv outputs/reports/accept_gate_audit_a.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return data


def _detect_kind(report: dict[str, Any]) -> str:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        return "unknown"
    if "frac_samples_never_visit_any_gold" in summary:
        return "accept_gate"
    if "frac_evidence_budget_saturated" in summary:
        return "evidence_saturation"
    return "unknown"


def _batch_label(report: dict[str, Any], path: Path) -> str:
    inputs = report.get("inputs")
    if isinstance(inputs, dict):
        bid = inputs.get("batch_id")
        if isinstance(bid, str) and bid:
            return bid
    return path.stem


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    if isinstance(x, int):
        return str(x)
    return str(x)


def _row_accept_gate(path: Path, report: dict[str, Any]) -> dict[str, str]:
    s = report.get("summary")
    if not isinstance(s, dict):
        s = {}
    return {
        "kind": "audit",
        "batch_id": _batch_label(report, path),
        "file": path.name,
        "n": _fmt(s.get("sample_count")),
        "never_visit": _fmt(s.get("frac_samples_never_visit_any_gold")),
        "visit_miss_accept": _fmt(s.get("frac_samples_visit_gold_but_missing_accept_for_some_visited_gold")),
        "all_visit_accepted": _fmt(s.get("frac_samples_with_visit_where_all_visited_gold_accepted")),
        "ctx_any_gold": _fmt(s.get("frac_samples_with_any_gold_in_context")),
        "mean_frac_gold_ctx": _fmt(s.get("mean_frac_gold_leaves_in_context")),
        "sum_acc_not_ctx": _fmt(s.get("sum_accepted_gold_not_in_context")),
    }


def _row_saturation(path: Path, report: dict[str, Any]) -> dict[str, str]:
    s = report.get("summary")
    if not isinstance(s, dict):
        s = {}
    return {
        "kind": "sat",
        "batch_id": _batch_label(report, path),
        "file": path.name,
        "n": _fmt(s.get("sample_count")),
        "sat_budget": _fmt(s.get("frac_evidence_budget_saturated")),
        "mean_n_evi": _fmt(s.get("mean_n_evidence"), 3),
        "gold_visit_dedup": _fmt(s.get("frac_gold_leaf_ever_visited_deduped")),
        "gold_in_accept": _fmt(s.get("frac_gold_in_accepted_evidence")),
        "gold_miss_evi": _fmt(s.get("sample_count_gold_missing_from_evidence")),
    }


def _print_table(rows: list[dict[str, str]], *, tsv: bool) -> None:
    if not rows:
        print("No rows.", file=sys.stderr)
        return
    keys = list(rows[0].keys())
    if tsv:
        print("\t".join(keys))
        for r in rows:
            print("\t".join(r.get(k, "") for k in keys))
        return
    widths = {k: max(len(k), max(len(r.get(k, "")) for r in rows)) for k in keys}
    sep = "  "
    header = sep.join(k.ljust(widths[k]) for k in keys)
    print(header)
    print(sep.join("-" * widths[k] for k in keys))
    for r in rows:
        print(sep.join(r.get(k, "").ljust(widths[k]) for k in keys))


def _collect_paths(paths: list[str], globs: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.append(Path(p))
    for pattern in globs:
        out.extend(sorted(Path(".").glob(pattern)))
    # de-dupe preserve order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="*", type=str, help="Diagnostic JSON files")
    p.add_argument("--glob", dest="globs", action="append", default=[], help="Glob (repeatable), cwd-relative")
    p.add_argument("--tsv", action="store_true", help="Tab-separated output")
    ns = p.parse_args(argv)

    files = _collect_paths(ns.paths, ns.globs)
    if not files:
        p.error("Provide at least one path or --glob")

    audit_rows: list[dict[str, str]] = []
    sat_rows: list[dict[str, str]] = []
    errors: list[str] = []

    for path in files:
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        try:
            report = _load(path)
            kind = _detect_kind(report)
            if kind == "accept_gate":
                audit_rows.append(_row_accept_gate(path, report))
            elif kind == "evidence_saturation":
                sat_rows.append(_row_saturation(path, report))
            else:
                errors.append(f"unknown report shape: {path}")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")

    if audit_rows:
        print("=== accept_gate_audit (summary) ===")
        _print_table(audit_rows, tsv=ns.tsv)
        print()
    if sat_rows:
        print("=== analyze_evidence_saturation (summary) ===")
        _print_table(sat_rows, tsv=ns.tsv)
        print()
    for e in errors:
        print(e, file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
