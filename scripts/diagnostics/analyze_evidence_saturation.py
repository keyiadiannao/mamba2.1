#!/usr/bin/env python3
"""
Aggregate evidence-budget saturation and gold-leaf reachability from run_payload.json files.

Typical server usage (from repo root):

  python scripts/diagnostics/analyze_evidence_saturation.py \\
    --glob 'outputs/runs/*/run_payload.json' \\
    --out-json outputs/reports/evidence_saturation_report.json

Or restrict to one batch via registry rows:

  python scripts/diagnostics/analyze_evidence_saturation.py \\
    --registry-jsonl outputs/reports/run_registry.jsonl \\
    --batch-id 'end_to_end_real_corpus_370m_qwen7b_rule_anticollapse_entityalpha_0.3_20260416_115828' \\
    --out-json outputs/reports/evidence_saturation_alpha03.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


def _extract_entity_key(node_id: str) -> str:
    if "__sent_" in node_id:
        return node_id.split("__sent_", 1)[0]
    return node_id


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return data


def analyze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    trace = payload.get("trace") or {}
    config = payload.get("config") or {}
    max_evidence = int(config.get("max_evidence", 3))
    evidence_texts = list(trace.get("evidence_texts") or [])
    evidence_node_ids = list(trace.get("evidence_node_ids") or [])
    n_evidence = len(evidence_texts)
    saturated = n_evidence >= max_evidence and max_evidence > 0

    entity_keys = [_extract_entity_key(nid) for nid in evidence_node_ids]
    unique_entities = len(set(entity_keys)) if entity_keys else 0
    same_entity_as_first = 0
    if entity_keys:
        first_e = entity_keys[0]
        same_entity_as_first = sum(1 for e in entity_keys if e == first_e)

    gold = {int(x) for x in (trace.get("leaf_indices_required") or []) if isinstance(x, int)}
    visited_deduped = set(trace.get("visited_leaf_indices_deduped") or [])
    visits_ordered = [int(x) for x in (trace.get("visited_leaf_visits_ordered") or []) if isinstance(x, int)]

    gold_index_first_in_visits: int | None = None
    if gold and visits_ordered:
        for idx, leaf_idx in enumerate(visits_ordered):
            if leaf_idx in gold:
                gold_index_first_in_visits = idx
                break

    gold_hit_visited = bool(gold & visited_deduped) if gold else None

    gold_in_accepted_evidence = False
    if gold:
        for event in trace.get("event_log") or []:
            if not isinstance(event, dict):
                continue
            if event.get("event") != "accept_evidence":
                continue
            li = event.get("leaf_index")
            if isinstance(li, int) and li in gold:
                gold_in_accepted_evidence = True
                break

    return {
        "run_id": payload.get("run_id"),
        "sample_id": payload.get("sample_id"),
        "batch_id": payload.get("batch_id"),
        "max_evidence": max_evidence,
        "n_evidence": n_evidence,
        "saturated": saturated,
        "unique_entities_in_evidence": unique_entities,
        "evidence_count_same_entity_as_first": same_entity_as_first,
        "n_gold_leaves": len(gold),
        "gold_hit_visited": gold_hit_visited,
        "gold_index_first_in_visits": gold_index_first_in_visits,
        "gold_in_accepted_evidence": gold_in_accepted_evidence if gold else None,
        "nav_success": trace.get("nav_success"),
        "exact_match": trace.get("exact_match"),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"sample_count": 0}

    with_gold = [r for r in rows if r.get("n_gold_leaves", 0) > 0]
    n_g = len(with_gold)

    def mean(key: str, subset: list[dict[str, Any]] | None = None) -> float:
        data = subset or rows
        vals = [float(r[key]) for r in data if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def frac(pred: Callable[[dict[str, Any]], bool], subset: list[dict[str, Any]] | None = None) -> float:
        data = subset or rows
        hits = sum(1 for r in data if pred(r))
        return hits / len(data) if data else 0.0

    summary: dict[str, Any] = {
        "sample_count": n,
        "sample_count_with_gold_annotation": n_g,
        "frac_evidence_budget_saturated": frac(lambda r: r.get("saturated")),
        "mean_n_evidence": mean("n_evidence"),
        "mean_max_evidence": mean("max_evidence"),
        "mean_unique_entities_in_evidence": mean("unique_entities_in_evidence"),
        "mean_evidence_same_entity_as_first": mean("evidence_count_same_entity_as_first"),
    }
    if n_g:
        summary["frac_gold_leaf_ever_visited_deduped"] = frac(
            lambda r: bool(r.get("gold_hit_visited")), with_gold
        )
        summary["frac_gold_in_accepted_evidence"] = frac(
            lambda r: bool(r.get("gold_in_accepted_evidence")), with_gold
        )
        first_hits = [r["gold_index_first_in_visits"] for r in with_gold if r.get("gold_index_first_in_visits") is not None]
        summary["mean_gold_index_first_in_visits"] = (
            sum(first_hits) / len(first_hits) if first_hits else None
        )
        missed = [r for r in with_gold if not r.get("gold_in_accepted_evidence")]
        summary["sample_count_gold_missing_from_evidence"] = len(missed)
        if missed:
            summary["frac_saturated_among_gold_missing"] = frac(lambda r: r.get("saturated"), missed)
    return summary


def _iter_payload_paths_from_glob(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if not paths and "*" not in pattern:
        p = root / pattern
        if p.is_file():
            paths = [p]
    return [p for p in paths if p.is_file()]


def _iter_payload_paths_from_registry(
    root: Path,
    registry_path: Path,
    batch_id: str | None,
    limit: int | None,
) -> list[Path]:
    rows_out: list[Path] = []
    with registry_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if batch_id and row.get("batch_id") != batch_id:
                continue
            out_dir = row.get("output_run_dir")
            if not out_dir:
                continue
            p = Path(out_dir)
            if not p.is_absolute():
                p = root / p
            candidate = p / "run_payload.json"
            if candidate.is_file():
                rows_out.append(candidate)
            if limit is not None and len(rows_out) >= limit:
                break
    return rows_out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--glob",
        dest="glob_pattern",
        help="Glob relative to --root, e.g. outputs/runs/*/run_payload.json",
    )
    src.add_argument(
        "--registry-jsonl",
        type=str,
        help="run_registry.jsonl: load run_payload.json from each row's output_run_dir",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="When using --registry-jsonl, only rows with this batch_id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max payloads to load (registry order)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repository root for relative paths (default: cwd)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Write full report (summary + per_sample) to this JSON file",
    )
    parser.add_argument(
        "--per-sample-csv",
        type=str,
        default=None,
        help="Optional CSV path for per-sample metrics",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()

    if args.glob_pattern:
        paths = _iter_payload_paths_from_glob(root, args.glob_pattern)
    else:
        reg = Path(args.registry_jsonl)
        if not reg.is_absolute():
            reg = root / reg
        paths = _iter_payload_paths_from_registry(root, reg, args.batch_id, args.limit)

    if not paths:
        print("No run_payload.json files matched.", file=sys.stderr)
        return 2

    per_sample: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            payload = _load_json(path)
            row = analyze_payload(payload)
            row["source_path"] = str(path)
            per_sample.append(row)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")

    report = {
        "summary": _summarize(per_sample),
        "per_sample": per_sample,
        "errors": errors,
        "inputs": {
            "root": str(root),
            "glob": args.glob_pattern,
            "registry_jsonl": args.registry_jsonl,
            "batch_id": args.batch_id,
            "payload_count": len(paths),
        },
    }

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if errors:
        print(f"\n{len(errors)} load errors (see report errors list).", file=sys.stderr)

    if args.out_json:
        out_path = Path(args.out_json)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote full report to {out_path}")

    if args.per_sample_csv:
        csv_path = Path(args.per_sample_csv)
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if per_sample:
            fieldnames = list(per_sample[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_sample)
            print(f"Wrote per-sample CSV to {csv_path}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
