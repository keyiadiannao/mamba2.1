#!/usr/bin/env python3
"""Build a single JSON digest from selected navigation run_payload.json files.

Use on the training server after a nav batch: point at sample_ids (or --preset curated),
glob ``outputs/runs/<run_prefix>_<sample_id>_*/run_payload.json``, and write one file you can
scp back for offline review (question head, gold leaves, visited dedup, route/event tails).

Usage::

  python scripts/diagnostics/export_nav_run_digests.py \\
    --root . \\
    --run-prefix nav_p0_visit_rule_entity_boost_a030 \\
    --preset curated \\
    --out-json outputs/reports/nav_digest_curated_122155Z.json

  # Or: merge bucket labels from the audit CSV export::
  python scripts/diagnostics/export_nav_run_digests.py \\
    --root . --run-prefix nav_p0_visit_rule_entity_boost_a030 \\
    --csv outputs/reports/audit_sample_buckets_nav_p0_visit_rule_entity_boost_a030_20260418_122155Z.csv \\
    --sample-id ce5bfc90086c11ebbd61ac1f6bf848b6 --sample-id 963fffd60bdd11eba7f7acde48001122 \\
    --out-json outputs/reports/nav_digest_pick.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Curated list from §6.6 follow-up: never_visit + visit_miss (cap / minrel).
CURATED_SAMPLE_IDS: list[str] = [
    "5b168cd8089811ebbd77ac1f6bf848b6",
    "c807422a0bda11eba7f7acde48001122",
    "b0e112940bdd11eba7f7acde48001122",
    "ad193b1808b411ebbd88ac1f6bf848b6",
    "7de9188c088b11ebbd6fac1f6bf848b6",
    "61456758085411ebbd59ac1f6bf848b6",
    "ce5bfc90086c11ebbd61ac1f6bf848b6",
    "346a41b4084e11ebbd56ac1f6bf848b6",
    "963fffd60bdd11eba7f7acde48001122",
    "5c9ef1d80bdd11eba7f7acde48001122",
    "922a629c0bb011ebab90acde48001122",
]


def _head_tail(seq: list[Any], head: int, tail: int) -> dict[str, Any]:
    n = len(seq)
    if n <= head + tail:
        return {"n": n, "all": seq}
    return {"n": n, "head": seq[:head], "tail": seq[-tail:]}


def _slim_route_decision(rd: dict[str, Any]) -> dict[str, Any]:
    cs = rd.get("child_scores") if isinstance(rd.get("child_scores"), list) else []
    slim_scores: list[dict[str, Any]] = []
    for x in cs[:6]:
        if not isinstance(x, dict):
            continue
        slim_scores.append(
            {
                "node_id": x.get("node_id"),
                "score": x.get("score"),
                "raw_router_score": x.get("raw_router_score"),
                "entity_match_score": x.get("entity_match_score"),
            }
        )
    oc = rd.get("ordered_child_ids") if isinstance(rd.get("ordered_child_ids"), list) else []
    return {
        "parent_node_id": rd.get("parent_node_id"),
        "depth": rd.get("depth"),
        "ordered_child_ids_head": [str(x) for x in oc[:10]],
        "child_scores_top6": slim_scores,
    }


def _digest_payload(
    payload: dict[str, Any],
    *,
    csv_row: dict[str, str] | None,
    route_head: int,
    route_tail: int,
    event_tail: int,
) -> dict[str, Any]:
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
    cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}

    visited = trace.get("visited_leaf_indices_deduped")
    if not isinstance(visited, list):
        visited = []
    gold = trace.get("leaf_indices_required")
    if not isinstance(gold, list):
        gold = []

    rds = trace.get("route_decisions") if isinstance(trace.get("route_decisions"), list) else []
    slim_rds = [_slim_route_decision(x) for x in rds if isinstance(x, dict)]
    rd_wrap = _head_tail(slim_rds, route_head, route_tail)

    evs = trace.get("event_log") if isinstance(trace.get("event_log"), list) else []
    ev_tail = evs[-event_tail:] if event_tail > 0 else []

    q = str(payload.get("question") or "")
    out: dict[str, Any] = {
        "run_id": payload.get("run_id"),
        "sample_id": payload.get("sample_id"),
        "batch_id": payload.get("batch_id"),
        "tree_path": payload.get("tree_path"),
        "question_head_240": q[:240],
        "leaf_indices_required_gold": gold,
        "visited_leaf_indices_deduped": visited,
        "visited_leaf_deduped_count": len(visited),
        "n_evidence": len(trace.get("evidence_texts") or []),
        "nav_wall_time_ms": trace.get("nav_wall_time_ms"),
        "failure_attribution": trace.get("failure_attribution"),
        "rollback_count": trace.get("rollback_count"),
        "config_nav_subset": {
            "max_nodes": cfg.get("max_nodes"),
            "max_depth": cfg.get("max_depth"),
            "max_evidence": cfg.get("max_evidence"),
            "min_relevance_score": cfg.get("min_relevance_score"),
            "entity_boost_alpha": cfg.get("entity_boost_alpha"),
            "explore_root_probe_top_m": cfg.get("explore_root_probe_top_m"),
            "explore_root_probe_budget_per_child": cfg.get("explore_root_probe_budget_per_child"),
        },
        "route_decisions": rd_wrap,
        "event_log_tail": ev_tail,
    }
    if csv_row:
        out["from_csv"] = {
            "bucket": csv_row.get("bucket"),
            "n_gold_leaves": csv_row.get("n_gold_leaves"),
            "n_gold_never_visited": csv_row.get("n_gold_never_visited"),
            "n_gold_visited_not_accepted_leaves": csv_row.get("n_gold_visited_not_accepted_leaves"),
            "visit_miss_dispositions": csv_row.get("visit_miss_dispositions"),
            "visited_leaf_deduped_count_csv": csv_row.get("visited_leaf_deduped_count"),
        }
    return out


def _find_payload(root: Path, run_prefix: str, sample_id: str) -> Path | None:
    pat = f"{run_prefix}_{sample_id}_*"
    matches = sorted(root.glob(f"outputs/runs/{pat}/run_payload.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _load_csv_index(csv_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("sample_id") or "")
            if sid:
                out[sid] = {k: str(v) if v is not None else "" for k, v in row.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, default=".", help="Repo root (default: cwd)")
    p.add_argument("--run-prefix", type=str, default="nav_p0_visit_rule_entity_boost_a030")
    p.add_argument("--csv", type=str, default=None, help="Optional audit_sample_buckets CSV for labels")
    p.add_argument("--preset", type=str, choices=("curated", ""), default="", help="Use built-in curated sample_id list")
    p.add_argument("--sample-id", action="append", default=[], dest="sample_ids", metavar="ID")
    p.add_argument("--ids-file", type=str, default=None, help="Text file: one sample_id per line")
    p.add_argument("--out-json", type=str, required=True)
    p.add_argument("--route-head", type=int, default=4)
    p.add_argument("--route-tail", type=int, default=8)
    p.add_argument("--event-tail", type=int, default=40)
    ns = p.parse_args(argv)

    root = Path(ns.root).resolve()
    ids: list[str] = list(ns.sample_ids)
    if ns.ids_file:
        text = Path(ns.ids_file).read_text(encoding="utf-8")
        ids.extend(line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))
    if ns.preset == "curated":
        ids.extend(CURATED_SAMPLE_IDS)
    # de-dupe preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for s in ids:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    if not uniq:
        print("No sample ids: use --preset curated or --sample-id / --ids-file", file=sys.stderr)
        return 2

    csv_index: dict[str, dict[str, str]] = {}
    if ns.csv:
        cp = Path(ns.csv)
        if not cp.is_file():
            print(f"CSV not found: {cp}", file=sys.stderr)
            return 2
        csv_index = _load_csv_index(cp.resolve())

    digests: list[dict[str, Any]] = []
    missing: list[str] = []
    for sid in uniq:
        path = _find_payload(root, ns.run_prefix, sid)
        if path is None or not path.is_file():
            missing.append(sid)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            missing.append(sid)
            continue
        row = csv_index.get(sid)
        digests.append(
            _digest_payload(
                payload,
                csv_row=row,
                route_head=int(ns.route_head),
                route_tail=int(ns.route_tail),
                event_tail=int(ns.event_tail),
            )
        )

    report = {
        "meta": {
            "root": str(root),
            "run_prefix": ns.run_prefix,
            "sample_ids_requested": uniq,
            "sample_ids_missing_payload": missing,
            "digest_count": len(digests),
            "csv": str(Path(ns.csv).resolve()) if ns.csv else None,
        },
        "digests": digests,
    }

    out_path = Path(ns.out_json)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(digests)} digests to {out_path}")
    if missing:
        print(f"Missing {len(missing)} sample_ids (no run_payload match): {missing[:20]}", file=sys.stderr)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
