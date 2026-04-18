#!/usr/bin/env python3
"""Summarize accept_gate_audit report per_sample into failure buckets.

With gold annotation only:
  never        — no gold leaf was ever visited (``gold_hit_visited`` false)
  visit_miss   — visited some gold but at least one visited gold not accepted
  visit_clean  — visited at least one gold and all visited gold in accept

Also prints mean ``n_gold_leaves`` per bucket and disposition totals for visit_miss rows.

Usage (from repo root):

  python scripts/diagnostics/summarize_audit_failure_buckets.py \\
    outputs/reports/accept_gate_audit_nav_p0_visit_rule_entity_boost_a030_....json
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit_json", type=str, help="Full report JSON from audit_accept_gate.py --out-json")
    ns = p.parse_args(argv)
    path = Path(ns.audit_json)
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 2

    data = _load(path)
    rows = data.get("per_sample")
    if not isinstance(rows, list):
        print("Missing per_sample list", file=sys.stderr)
        return 2

    with_gold = [r for r in rows if isinstance(r, dict) and int(r.get("n_gold_leaves") or 0) > 0]
    n = len(with_gold)
    if not n:
        print("No samples with gold annotation.")
        return 0

    never = [r for r in with_gold if not r.get("gold_hit_visited")]
    visit_miss = [
        r
        for r in with_gold
        if r.get("gold_hit_visited") and int(r.get("n_gold_visited_not_accepted_leaves") or 0) > 0
    ]
    visit_clean = [
        r
        for r in with_gold
        if r.get("gold_hit_visited")
        and int(r.get("n_gold_visited_not_accepted_leaves") or 0) == 0
        and r.get("all_visited_gold_in_accepted") is True
    ]

    def mean_gold(sub: list[dict[str, Any]]) -> float:
        if not sub:
            return 0.0
        return sum(int(r.get("n_gold_leaves") or 0) for r in sub) / float(len(sub))

    disp_tot: dict[str, int] = {}
    for r in visit_miss:
        dmap = r.get("visited_not_accepted_dispositions") or {}
        if not isinstance(dmap, dict):
            continue
        for k, v in dmap.items():
            if isinstance(v, int):
                disp_tot[str(k)] = disp_tot.get(str(k), 0) + v

    bid = None
    inp = data.get("inputs")
    if isinstance(inp, dict):
        bid = inp.get("batch_id")

    print(f"batch_id\t{bid or '(unknown)'}")
    print(f"with_gold_n\t{n}")
    print(f"never_visit_n\t{len(never)}\tfrac\t{len(never) / float(n):.4f}\tmean_n_gold\t{mean_gold(never):.3f}")
    print(f"visit_miss_n\t{len(visit_miss)}\tfrac\t{len(visit_miss) / float(n):.4f}\tmean_n_gold\t{mean_gold(visit_miss):.3f}")
    print(f"visit_clean_n\t{len(visit_clean)}\tfrac\t{len(visit_clean) / float(n):.4f}\tmean_n_gold\t{mean_gold(visit_clean):.3f}")
    if disp_tot:
        print("visit_miss_dispositions_leaf_events\t" + json.dumps(dict(sorted(disp_tot.items())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
