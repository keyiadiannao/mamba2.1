from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase A SSGS pipeline.")
    parser.add_argument(
        "--config",
        default="configs/experiment/phase_a_demo.json",
        help="Path to the experiment config JSON.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    config_path = ROOT / Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tree_path = ROOT / str(config["tree_path"])
    tree_payload = load_tree_payload(tree_path)
    tree = load_tree_from_json(tree_path)

    question = str(config.get("question") or tree_payload.get("question") or "")
    if not question:
        raise ValueError("A question must be provided in the config or tree payload.")

    controller = SSGSController(
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
    trace = controller.run(question, tree)
    prompt = build_generator_prompt(question, trace.evidence_texts)

    reference_answer = tree_payload.get("reference_answer")
    if isinstance(reference_answer, str) and trace.evidence_texts:
        trace.exact_match = exact_match(trace.evidence_texts[0], reference_answer)

    run_id = make_run_id(str(config.get("run_id_prefix", "phase_a")))
    output_dir = ROOT / str(config.get("output_dir", "outputs/runs"))
    payload = {
        "run_id": run_id,
        "config": config,
        "question": question,
        "tree_path": str(config["tree_path"]),
        "trace": trace.to_dict(),
        "generator_prompt": prompt,
        "reference_answer": reference_answer,
    }

    output_path = write_run_payload(output_dir, payload, run_id)
    payload["output_run_dir"] = str(output_path.parent)
    write_json(output_path, payload)
    registry_row = build_registry_row(payload)
    navigation_summary = build_navigation_summary(payload)
    write_json(output_path.parent / "registry_row.json", registry_row)
    write_json(output_path.parent / "navigation_summary.json", navigation_summary)
    registry_path = append_jsonl(ROOT / "outputs" / "reports" / "run_registry.jsonl", registry_row)
    navigation_registry_path = append_jsonl(
        ROOT / "outputs" / "reports" / "navigation_summary.jsonl",
        navigation_summary,
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved run payload to: {output_path}")
    print(f"Saved registry row to: {output_path.parent / 'registry_row.json'}")
    print(f"Saved navigation summary to: {output_path.parent / 'navigation_summary.json'}")
    print(f"Updated run registry: {registry_path}")
    print(f"Updated navigation summary registry: {navigation_registry_path}")


if __name__ == "__main__":
    main()
