#!/usr/bin/env python3
"""Summarize navigation ``event_log`` signals for **gold-but-never-visited** samples.

Reads ``accept_gate_audit_*.json`` (from ``audit_accept_gate.py --out-json``), selects rows
with ``n_gold_leaves > 0`` and ``gold_hit_visited == false``, loads each ``source_path``
``run_payload.json``, and aggregates:

- fraction of rows whose trace contains ``max_nodes_reached`` / ``max_depth_reached`` events
- mean ``len(visited_node_ids)``, ``len(evidence_texts)``, ``rollback_count``, ``snapshot_stack_max_depth``

Use after a navigation batch to see whether ``never_visit`` is dominated by **budget caps**
vs **routing / ordering** (caps often absent when ``max_nodes`` is not the stopper).

Usage (repo root):

  python scripts/diagnostics/summarize_never_visit_nav_signals.py \\
    --audit-json outputs/reports/accept_gate_audit_<batch>.json \\
    --root .
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


def _event_counts(events: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        k = str(ev.get("event") or "")
        if not k:
            continue
        out[k] = out.get(k, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-json", type=str, required=True)
    p.add_argument("--root", type=str, default=".", help="Resolve relative source_path / cwd")
    ns = p.parse_args(argv)
    root = Path(ns.root).resolve()
    audit_path = Path(ns.audit_json)
    if not audit_path.is_absolute():
        audit_path = root / audit_path
    if not audit_path.is_file():
        print(f"Not found: {audit_path}", file=sys.stderr)
        return 2

    report = _load(audit_path)
    rows = report.get("per_sample")
    if not isinstance(rows, list):
        print("Missing per_sample list", file=sys.stderr)
        return 2

    never: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if int(r.get("n_gold_leaves") or 0) <= 0:
            continue
        if r.get("gold_hit_visited") is True:
            continue
        never.append(r)

    n = len(never)
    if n == 0:
        print(json.dumps({"never_visit_gold_rows": 0, "note": "no rows"}, indent=2, ensure_ascii=False))
        return 0

    with_max_nodes = 0
    with_max_depth = 0
    with_root_probe = 0
    missing_path = 0
    load_errors = 0

    sum_nodes = 0
    sum_evidence = 0
    sum_roll = 0
    sum_snap_depth = 0
    sum_leaf_dedup = 0

    for r in never:
        sp = r.get("source_path")
        if not isinstance(sp, str) or not sp.strip():
            missing_path += 1
            continue
        ppath = Path(sp)
        if not ppath.is_absolute():
            ppath = root / ppath
        if not ppath.is_file():
            missing_path += 1
            continue
        try:
            payload = _load(ppath)
        except (OSError, json.JSONDecodeError, ValueError):
            load_errors += 1
            continue
        trace = payload.get("trace")
        if not isinstance(trace, dict):
            load_errors += 1
            continue
        events = trace.get("event_log") if isinstance(trace.get("event_log"), list) else []
        ec = _event_counts(events)
        if ec.get("max_nodes_reached", 0) > 0:
            with_max_nodes += 1
        if ec.get("max_depth_reached", 0) > 0:
            with_max_depth += 1
        if ec.get("root_probe_plan", 0) > 0:
            with_root_probe += 1

        vids = trace.get("visited_node_ids")
        sum_nodes += len(vids) if isinstance(vids, list) else 0
        evs = trace.get("evidence_texts")
        sum_evidence += len(evs) if isinstance(evs, list) else 0
        sum_roll += int(trace.get("rollback_count") or 0)
        sum_snap_depth += int(trace.get("snapshot_stack_max_depth") or 0)
        ddup = trace.get("visited_leaf_indices_deduped")
        sum_leaf_dedup += len(ddup) if isinstance(ddup, list) else 0

    def frac(x: int) -> float:
        return float(x) / float(n) if n else 0.0

    out = {
        "audit_json": str(audit_path),
        "never_visit_gold_rows": n,
        "resolved_payloads": n - missing_path - load_errors,
        "missing_or_bad_source_path": missing_path,
        "payload_load_errors": load_errors,
        "frac_never_visit_with_event_max_nodes_reached": round(frac(with_max_nodes), 4),
        "frac_never_visit_with_event_max_depth_reached": round(frac(with_max_depth), 4),
        "frac_never_visit_with_event_root_probe_plan": round(frac(with_root_probe), 4),
        "mean_visited_node_ids_len": round(sum_nodes / float(n), 3),
        "mean_evidence_texts_len": round(sum_evidence / float(n), 3),
        "mean_rollback_count": round(sum_roll / float(n), 3),
        "mean_snapshot_stack_max_depth": round(sum_snap_depth / float(n), 3),
        "mean_visited_leaf_indices_deduped_len": round(sum_leaf_dedup / float(n), 3),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(
        "\nRead: never_visit rows=%d | max_nodes_reached in %.1f%% | max_depth_reached in %.1f%%"
        % (n, 100.0 * frac(with_max_nodes), 100.0 * frac(with_max_depth)),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
