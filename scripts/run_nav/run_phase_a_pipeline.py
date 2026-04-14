from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import load_json, run_navigation_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase A SSGS pipeline.")
    parser.add_argument(
        "--config",
        default="configs/experiment/phase_a_demo.json",
        help="Path to the experiment config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / Path(args.config))
    payload = run_navigation_sample(
        root_dir=ROOT,
        config=config,
        question=str(config.get("question") or ""),
        tree_path=str(config["tree_path"]),
        reference_answer=str(config["reference_answer"]) if config.get("reference_answer") else None,
        run_id_prefix=str(config.get("run_id_prefix", "phase_a")),
        sample_id=str(config.get("sample_id")) if config.get("sample_id") else None,
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved run payload to: {Path(payload['output_run_dir']) / 'run_payload.json'}")
    print(f"Saved registry row to: {Path(payload['output_run_dir']) / 'registry_row.json'}")
    print(f"Saved navigation summary to: {Path(payload['output_run_dir']) / 'navigation_summary.json'}")
    print(f"Updated run registry: {ROOT / 'outputs' / 'reports' / 'run_registry.jsonl'}")
    print(f"Updated navigation summary registry: {ROOT / 'outputs' / 'reports' / 'navigation_summary.jsonl'}")


if __name__ == "__main__":
    main()
