from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import build_batch_summary, build_controller, load_json, run_navigation_sample
from src.tracing import append_jsonl, make_run_id, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end evaluation batch.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the end-to-end batch experiment config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / Path(args.config))
    if not bool(config.get("run_generator", False)):
        raise ValueError("End-to-end batch config must set `run_generator` to true.")

    samples_path = ROOT / str(config["samples_path"])
    samples_payload = load_json(samples_path)
    samples = list(samples_payload.get("samples", []))
    if not samples:
        raise ValueError("End-to-end batch config requires a non-empty samples list.")

    batch_id = make_run_id(str(config.get("batch_id_prefix", "end_to_end_batch")))
    controller = build_controller(config)
    sample_payloads = []
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id", f"sample_{index:03d}"))
        payload = run_navigation_sample(
            root_dir=ROOT,
            config=config,
            question=str(sample.get("question") or config.get("question") or ""),
            tree_path=str(sample.get("tree_path") or config["tree_path"]),
            reference_answer=str(sample["reference_answer"]) if sample.get("reference_answer") else None,
            run_id_prefix=f"{config.get('run_id_prefix', 'end_to_end')}_{sample_id}",
            sample_id=sample_id,
            batch_id=batch_id,
            leaf_indices_required=list(sample.get("positive_leaf_indices", [])),
            controller=controller,
        )
        sample_payloads.append(payload)

    batch_summary = build_batch_summary(batch_id, sample_payloads)
    batch_summary["config"] = config
    batch_summary["samples_path"] = str(config["samples_path"])

    batch_output_dir = ROOT / str(config.get("batch_output_dir", "outputs/reports/end_to_end_batches"))
    write_json(batch_output_dir / batch_id / "batch_summary.json", batch_summary)
    append_jsonl(ROOT / "outputs" / "reports" / "end_to_end_batch_summary.jsonl", batch_summary)

    print(json.dumps(batch_summary, indent=2, ensure_ascii=False))
    print(f"\nSaved batch summary to: {batch_output_dir / batch_id / 'batch_summary.json'}")
    print(f"Updated batch summary registry: {ROOT / 'outputs' / 'reports' / 'end_to_end_batch_summary.jsonl'}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import build_batch_summary, build_controller, load_json, run_navigation_sample
from src.tracing import append_jsonl, make_run_id, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end evaluation batch.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the end-to-end batch experiment config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / Path(args.config))
    if not bool(config.get("run_generator", False)):
        raise ValueError("End-to-end batch config must set `run_generator` to true.")

    samples_path = ROOT / str(config["samples_path"])
    samples_payload = load_json(samples_path)
    samples = list(samples_payload.get("samples", []))
    if not samples:
        raise ValueError("End-to-end batch config requires a non-empty samples list.")

    batch_id = make_run_id(str(config.get("batch_id_prefix", "end_to_end_batch")))
    controller = build_controller(config)
    sample_payloads = []
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id", f"sample_{index:03d}"))
        payload = run_navigation_sample(
            root_dir=ROOT,
            config=config,
            question=str(sample.get("question") or config.get("question") or ""),
            tree_path=str(sample.get("tree_path") or config["tree_path"]),
            reference_answer=str(sample["reference_answer"]) if sample.get("reference_answer") else None,
            run_id_prefix=f"{config.get('run_id_prefix', 'end_to_end')}_{sample_id}",
            sample_id=sample_id,
            batch_id=batch_id,
            leaf_indices_required=list(sample.get("positive_leaf_indices", [])),
            controller=controller,
        )
        sample_payloads.append(payload)

    batch_summary = build_batch_summary(batch_id, sample_payloads)
    batch_summary["config"] = config
    batch_summary["samples_path"] = str(config["samples_path"])

    batch_output_dir = ROOT / str(config.get("batch_output_dir", "outputs/reports/end_to_end_batches"))
    write_json(batch_output_dir / batch_id / "batch_summary.json", batch_summary)
    append_jsonl(ROOT / "outputs" / "reports" / "end_to_end_batch_summary.jsonl", batch_summary)

    print(json.dumps(batch_summary, indent=2, ensure_ascii=False))
    print(f"\nSaved batch summary to: {batch_output_dir / batch_id / 'batch_summary.json'}")
    print(f"Updated batch summary registry: {ROOT / 'outputs' / 'reports' / 'end_to_end_batch_summary.jsonl'}")


if __name__ == "__main__":
    main()
