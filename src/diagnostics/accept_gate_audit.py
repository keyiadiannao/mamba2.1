"""Offline audit: gold leaf visit vs accept_evidence (Controller gate vs read-side).

Optional: when ``root_dir`` is passed to ``audit_payload``, also compares gold leaves
to ``trace.context_node_ids`` (generator context after ``context_select_*``).

See ``scripts/diagnostics/audit_accept_gate.py`` for CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _gold_set(trace: dict[str, Any]) -> set[int]:
    return {int(x) for x in (trace.get("leaf_indices_required") or []) if isinstance(x, int)}


def _leaf_index_to_node_id_map(root: Path, tree_path: str) -> dict[int, str] | None:
    """Return leaf_index -> node_id for leaves that carry ``leaf_index`` metadata."""
    path = (root / tree_path).resolve()
    if not path.is_file():
        return None
    try:
        from src.tree_builder import load_tree_from_payload, load_tree_payload

        tree = load_tree_from_payload(load_tree_payload(path))
    except (OSError, ValueError, TypeError, KeyError):
        return None
    out: dict[int, str] = {}
    for node in tree.walk_depth_first():
        if not node.is_leaf:
            continue
        li = node.metadata.get("leaf_index")
        if isinstance(li, int):
            out[int(li)] = str(node.node_id)
    return out


def _compute_context_gold_metrics(
    gold: set[int],
    accepted_gold: set[int],
    leaf_index_to_node_id: dict[int, str],
    context_node_ids: list[str] | None,
) -> dict[str, Any]:
    ctx_ids = set(context_node_ids or [])
    gold_in_context = {li for li in gold if leaf_index_to_node_id.get(li) in ctx_ids}
    n_in = len(gold_in_context)
    n_gold = len(gold)
    return {
        "n_gold_leaves_in_context": n_in,
        "frac_gold_leaves_in_context": (float(n_in) / float(n_gold)) if n_gold else None,
        "n_accepted_gold_not_in_context": len(accepted_gold - gold_in_context),
    }


def _accepted_gold_leaf_indices(trace: dict[str, Any], gold: set[int]) -> set[int]:
    out: set[int] = set()
    for ev in trace.get("event_log") or []:
        if not isinstance(ev, dict) or ev.get("event") != "accept_evidence":
            continue
        li = ev.get("leaf_index")
        if isinstance(li, int) and li in gold:
            out.add(li)
    return out


def _leaf_disposition_from_events(trace: dict[str, Any], leaf_index: int) -> str:
    """Best-effort reason why a visited gold leaf did not end in accept_evidence (or accepted)."""
    events = [e for e in (trace.get("event_log") or []) if isinstance(e, dict) and e.get("leaf_index") == leaf_index]
    if not events:
        return "no_leaf_index_events"

    kinds = [str(e.get("event") or "") for e in events]
    if any(k == "accept_evidence" for k in kinds):
        return "accepted"

    # Prefer the last decisive leaf-level event
    for ev in reversed(events):
        k = str(ev.get("event") or "")
        if k == "reject_leaf":
            return "reject_leaf_min_relevance"
        if k == "reject_leaf_branch_cap":
            return "reject_leaf_branch_cap"
        if k == "skip_duplicate_evidence":
            return "skip_duplicate_evidence"

    return "unknown_leaf_events"


def audit_trace(trace: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return per-sample audit row (no I/O)."""
    cfg = config or {}
    gold = _gold_set(trace)
    visited_deduped = {int(x) for x in (trace.get("visited_leaf_indices_deduped") or []) if isinstance(x, int)}
    visits_ordered = [int(x) for x in (trace.get("visited_leaf_visits_ordered") or []) if isinstance(x, int)]

    accepted_gold = _accepted_gold_leaf_indices(trace, gold)
    visited_gold = gold & visited_deduped
    never_visited_gold = gold - visited_deduped
    visited_not_accepted = visited_gold - accepted_gold

    n_gold = len(gold)
    n_never_visited = len(never_visited_gold)
    n_visited_not_accepted = len(visited_not_accepted)

    dispositions: dict[str, int] = {}
    for li in sorted(visited_not_accepted):
        d = _leaf_disposition_from_events(trace, li)
        dispositions[d] = dispositions.get(d, 0) + 1

    gold_hit_visited = bool(visited_gold) if gold else False

    return {
        "n_gold_leaves": n_gold,
        "n_gold_never_visited": n_never_visited,
        "n_gold_visited_not_accepted_leaves": n_visited_not_accepted,
        "gold_hit_visited": gold_hit_visited if gold else None,
        "all_visited_gold_in_accepted": (bool(visited_gold) and visited_gold <= accepted_gold) if gold else None,
        "visited_not_accepted_dispositions": dispositions,
        "max_evidence_config": int(cfg.get("max_evidence", 3)),
        "min_relevance_score_config": float(cfg.get("min_relevance_score", 1.0)),
        "n_evidence": len(trace.get("evidence_texts") or []),
        "visited_leaf_visit_count": len(visits_ordered),
        "visited_leaf_deduped_count": len(visited_deduped),
    }


def audit_payload(payload: dict[str, Any], *, root_dir: Path | None = None) -> dict[str, Any]:
    trace = payload.get("trace") or {}
    config = payload.get("config") or {}
    row = audit_trace(trace, config=config)
    row["run_id"] = payload.get("run_id")
    row["sample_id"] = payload.get("sample_id")
    row["batch_id"] = payload.get("batch_id")

    gold = _gold_set(trace)
    accepted_gold = _accepted_gold_leaf_indices(trace, gold)
    row["n_gold_leaves_accepted"] = len(accepted_gold)
    row["frac_gold_leaves_accepted"] = (float(len(accepted_gold)) / float(len(gold))) if gold else None

    row["context_gold_metrics_available"] = False
    row["context_gold_metrics_skip_reason"] = None
    if not gold:
        row["n_gold_leaves_in_context"] = 0
        row["frac_gold_leaves_in_context"] = None
        row["n_accepted_gold_not_in_context"] = 0
        return row

    if root_dir is None:
        row["context_gold_metrics_skip_reason"] = "root_dir_not_provided"
        row["n_gold_leaves_in_context"] = None
        row["frac_gold_leaves_in_context"] = None
        row["n_accepted_gold_not_in_context"] = None
        return row

    tree_path = str(payload.get("tree_path") or "")
    if not tree_path:
        row["context_gold_metrics_skip_reason"] = "payload_missing_tree_path"
        row["n_gold_leaves_in_context"] = None
        row["frac_gold_leaves_in_context"] = None
        row["n_accepted_gold_not_in_context"] = None
        return row

    mapping = _leaf_index_to_node_id_map(root_dir, tree_path)
    if not mapping:
        row["context_gold_metrics_skip_reason"] = "tree_unreadable_or_missing"
        row["n_gold_leaves_in_context"] = None
        row["frac_gold_leaves_in_context"] = None
        row["n_accepted_gold_not_in_context"] = None
        return row

    ctx_metrics = _compute_context_gold_metrics(
        gold,
        accepted_gold,
        mapping,
        trace.get("context_node_ids") if isinstance(trace.get("context_node_ids"), list) else [],
    )
    row.update(ctx_metrics)
    row["context_gold_metrics_available"] = True
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    with_gold = [r for r in rows if int(r.get("n_gold_leaves") or 0) > 0]
    n_g = len(with_gold)

    def frac(pred: Any) -> float:
        if not with_gold:
            return 0.0
        return sum(1 for r in with_gold if pred(r)) / float(len(with_gold))

    agg_disp: dict[str, int] = {}
    for r in with_gold:
        dmap = r.get("visited_not_accepted_dispositions") or {}
        if not isinstance(dmap, dict):
            continue
        for k, v in dmap.items():
            if isinstance(v, int):
                agg_disp[k] = agg_disp.get(k, 0) + v

    with_visit = [r for r in with_gold if r.get("gold_hit_visited")]

    def frac_sub(sub: list[dict[str, Any]], pred: Any) -> float:
        if not sub:
            return 0.0
        return sum(1 for r in sub if pred(r)) / float(len(sub))

    ctx_ok = [r for r in with_gold if r.get("context_gold_metrics_available")]

    def frac_ctx(pred: Any) -> float:
        if not ctx_ok:
            return 0.0
        return sum(1 for r in ctx_ok if pred(r)) / float(len(ctx_ok))

    mean_frac_in_context = 0.0
    if ctx_ok:
        vals = [float(r["frac_gold_leaves_in_context"]) for r in ctx_ok if r.get("frac_gold_leaves_in_context") is not None]
        mean_frac_in_context = sum(vals) / float(len(vals)) if vals else 0.0

    return {
        "sample_count": n,
        "sample_count_with_gold_annotation": n_g,
        "frac_samples_never_visit_any_gold": frac(lambda r: not r.get("gold_hit_visited")),
        "frac_samples_visit_gold_but_missing_accept_for_some_visited_gold": frac(
            lambda r: int(r.get("n_gold_visited_not_accepted_leaves") or 0) > 0
        ),
        "frac_samples_with_visit_where_all_visited_gold_accepted": frac_sub(
            with_visit, lambda r: r.get("all_visited_gold_in_accepted") is True
        ),
        "sum_gold_leaves_never_visited": sum(int(r.get("n_gold_never_visited") or 0) for r in with_gold),
        "sum_gold_leaves_visited_not_accepted": sum(int(r.get("n_gold_visited_not_accepted_leaves") or 0) for r in with_gold),
        "visited_not_accepted_dispositions_aggregated": agg_disp,
        "context_gold_metrics_sample_count": len(ctx_ok),
        "frac_samples_with_any_gold_in_context": frac_ctx(lambda r: int(r.get("n_gold_leaves_in_context") or 0) > 0),
        "mean_frac_gold_leaves_in_context": mean_frac_in_context,
        "sum_accepted_gold_not_in_context": sum(
            int(r.get("n_accepted_gold_not_in_context") or 0) for r in ctx_ok
        ),
    }
