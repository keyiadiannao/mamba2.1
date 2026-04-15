from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.controller import ControllerConfig, SSGSController
from src.evaluation import exact_match
from src.generator_bridge import build_generator_prompt
from src.navigator import build_navigator
from src.router import build_router
from src.tracing import (
    append_jsonl,
    build_navigation_summary,
    build_registry_row,
    make_run_id,
    write_json,
    write_run_payload,
)
from src.tree_builder import load_tree_from_json, load_tree_payload


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_controller(config: dict[str, Any]) -> SSGSController:
    return SSGSController(
        navigator=build_navigator(config),
        router=build_router(config),
        config=ControllerConfig(
            routing_mode=str(config.get("routing_mode", "rule")),
            context_source=str(config.get("context_source", "t1_visited_leaves_ordered")),
            max_evidence=int(config.get("max_evidence", 3)),
            min_relevance_score=float(config.get("min_relevance_score", 1.0)),
            max_depth=int(config.get("max_depth", 8)),
            max_nodes=int(config.get("max_nodes", 64)),
        ),
    )


def run_navigation_sample(
    root_dir: Path,
    config: dict[str, Any],
    question: str,
    tree_path: str,
    reference_answer: str | None = None,
    run_id_prefix: str | None = None,
    sample_id: str | None = None,
    leaf_indices_required: list[int] | None = None,
) -> dict[str, Any]:
    resolved_tree_path = root_dir / tree_path
    tree_payload = load_tree_payload(resolved_tree_path)
    tree = load_tree_from_json(resolved_tree_path)

    final_question = question or str(tree_payload.get("question") or "")
    if not final_question:
        raise ValueError("A question must be provided in the config, sample, or tree payload.")

    controller = build_controller(config)
    trace = controller.run(final_question, tree)
    if leaf_indices_required is not None:
        trace.leaf_indices_required = list(leaf_indices_required)
    prompt = build_generator_prompt(final_question, trace.evidence_texts)

    final_reference = reference_answer
    if final_reference is None:
        tree_reference = tree_payload.get("reference_answer")
        final_reference = tree_reference if isinstance(tree_reference, str) else None

    if isinstance(final_reference, str) and trace.evidence_texts:
        trace.exact_match = exact_match(trace.evidence_texts[0], final_reference)

    run_id = make_run_id(run_id_prefix or str(config.get("run_id_prefix", "phase_a")))
    payload = {
        "run_id": run_id,
        "sample_id": sample_id,
        "config": config,
        "question": final_question,
        "tree_path": tree_path,
        "trace": trace.to_dict(),
        "generator_prompt": prompt,
        "reference_answer": final_reference,
    }

    output_dir = root_dir / str(config.get("output_dir", "outputs/runs"))
    output_path = write_run_payload(output_dir, payload, run_id)
    payload["output_run_dir"] = str(output_path.parent)
    write_json(output_path, payload)

    registry_row = build_registry_row(payload)
    navigation_summary = build_navigation_summary(payload)
    write_json(output_path.parent / "registry_row.json", registry_row)
    write_json(output_path.parent / "navigation_summary.json", navigation_summary)
    append_jsonl(root_dir / "outputs" / "reports" / "run_registry.jsonl", registry_row)
    append_jsonl(root_dir / "outputs" / "reports" / "navigation_summary.jsonl", navigation_summary)

    return payload


def build_batch_summary(batch_id: str, sample_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(sample_payloads)
    nav_successes = sum(1 for payload in sample_payloads if payload["trace"].get("nav_success"))
    exact_match_hits = sum(
        1
        for payload in sample_payloads
        if payload["trace"].get("exact_match") == 1
    )
    total_nav_time = sum(float(payload["trace"].get("nav_wall_time_ms") or 0.0) for payload in sample_payloads)

    return {
        "batch_id": batch_id,
        "sample_count": sample_count,
        "nav_success_count": nav_successes,
        "nav_success_rate": (nav_successes / sample_count) if sample_count else 0.0,
        "exact_match_count": exact_match_hits,
        "exact_match_rate": (exact_match_hits / sample_count) if sample_count else 0.0,
        "avg_nav_wall_time_ms": (total_nav_time / sample_count) if sample_count else 0.0,
        "sample_run_ids": [payload["run_id"] for payload in sample_payloads],
        "sample_ids": [payload.get("sample_id") for payload in sample_payloads],
    }
