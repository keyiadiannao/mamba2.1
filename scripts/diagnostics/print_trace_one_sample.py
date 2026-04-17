#!/usr/bin/env python3
"""
Run a single end-to-end sample (same path as batch) and print a compact trace to stdout.

Writes under a temp directory only (default), so outputs/runs is not flooded.

Examples (repo root):

  python scripts/diagnostics/print_trace_one_sample.py \\
    --config configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_rule.example.json \\
    --sample-index 0 --no-generator

  # Real GPU nav + Qwen (slow); omit --no-generator and set GENERATOR_HF_MODEL_NAME or --generator-hf-model-name

  python scripts/diagnostics/print_trace_one_sample.py \\
    --config configs/experiment/end_to_end_batch_demo_smoke_ctxsel_overlap_k4.json \\
    --sample-id einstein_relativity --no-generator
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline import load_json, run_navigation_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to batch experiment JSON (relative to repo root).")
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index into samples list (ignored if --sample-id set).",
    )
    p.add_argument("--sample-id", default=None, help="Pick sample by sample_id.")
    p.add_argument(
        "--no-generator",
        action="store_true",
        help="Force run_generator=false for fast trace-only print.",
    )
    p.add_argument(
        "--generator-hf-model-name",
        default=None,
        help="Override generator_hf_model_name (same as run_end_to_end_batch).",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Print temp dir path and do not delete (for inspecting run_payload on disk).",
    )
    p.add_argument(
        "--event-log-max",
        type=int,
        default=0,
        help="If >0, only first N events of event_log (avoid huge paste). 0 = full log.",
    )
    return p.parse_args()


def _pick_sample(samples: list[dict], sample_id: str | None, index: int) -> dict:
    if sample_id:
        for row in samples:
            if str(row.get("sample_id")) == sample_id:
                return row
        raise SystemExit(f"No sample_id={sample_id!r} in batch.")
    if index < 0 or index >= len(samples):
        raise SystemExit(f"sample-index {index} out of range (n={len(samples)}).")
    return samples[index]


def _gold_in_accepted(trace: dict) -> bool:
    gold = {int(x) for x in (trace.get("leaf_indices_required") or []) if isinstance(x, int)}
    if not gold:
        return False
    for ev in trace.get("event_log") or []:
        if not isinstance(ev, dict) or ev.get("event") != "accept_evidence":
            continue
        li = ev.get("leaf_index")
        if isinstance(li, int) and li in gold:
            return True
    return False


def main() -> int:
    args = parse_args()
    cfg_path = _REPO_ROOT / Path(args.config)
    base = load_json(cfg_path)
    td = Path(tempfile.mkdtemp(prefix="trace_one_"))
    try:
        cfg = dict(base)
        cfg["output_dir"] = str(td / "runs")
        cfg["report_dir"] = str(td / "reports")
        if args.no_generator:
            cfg["run_generator"] = False
        gen = (args.generator_hf_model_name or os.environ.get("GENERATOR_HF_MODEL_NAME") or "").strip()
        if gen:
            cfg["generator_hf_model_name"] = gen

        samples_path = _REPO_ROOT / str(cfg["samples_path"])
        batch = load_json(samples_path)
        samples = list(batch.get("samples", []))
        if not samples:
            raise SystemExit("samples list is empty.")
        sample = _pick_sample(samples, args.sample_id, args.sample_index)

        question = str(sample.get("question") or cfg.get("question") or "")
        tree_path = str(sample.get("tree_path") or cfg["tree_path"])
        ref = sample.get("reference_answer")
        gold = list(sample.get("positive_leaf_indices", []))

        payload = run_navigation_sample(
            root_dir=_REPO_ROOT,
            config=cfg,
            question=question,
            tree_path=tree_path,
            reference_answer=str(ref) if ref else None,
            run_id_prefix="diag_print_trace_one",
            sample_id=str(sample.get("sample_id", "sample")),
            batch_id="diag_print_trace_one_batch",
            leaf_indices_required=gold if gold else None,
        )

        trace = payload["trace"]
        events = trace.get("event_log") or []
        events_out = events
        if args.event_log_max and len(events) > args.event_log_max:
            events_out = events[: args.event_log_max]
        ctr = Counter(str(e.get("event")) for e in events if isinstance(e, dict))

        slim = {
            "sample_id": payload.get("sample_id"),
            "tree_path": tree_path,
            "nav_keys": {
                "nav_success": trace.get("nav_success"),
                "n_evidence": len(trace.get("evidence_texts") or []),
                "max_evidence": cfg.get("max_evidence"),
                "min_relevance_score": cfg.get("min_relevance_score"),
                "max_depth": cfg.get("max_depth"),
                "max_nodes": cfg.get("max_nodes"),
                "routing_mode": trace.get("routing_mode", cfg.get("routing_mode")),
                "rollback_count": trace.get("rollback_count"),
            },
            "gold": {
                "leaf_indices_required": trace.get("leaf_indices_required"),
                "gold_in_accepted_evidence": _gold_in_accepted(trace),
                "visited_leaf_indices_deduped": trace.get("visited_leaf_indices_deduped"),
                "n_visited_ordered": len(trace.get("visited_leaf_visits_ordered") or []),
            },
            "event_counts": dict(ctr),
            "event_log": events_out,
            "event_log_truncated": bool(args.event_log_max and len(events) > args.event_log_max),
            "event_log_total": len(events),
        }
        print(json.dumps(slim, indent=2, ensure_ascii=False))
        print("\n--- temp dir (run_payload + registry_row) ---", file=sys.stderr)
        print(str(payload.get("output_run_dir", td)), file=sys.stderr)
        if args.keep_temp:
            print(f"KEPT temp dir: {td}", file=sys.stderr)
        return 0
    finally:
        if not args.keep_temp:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
