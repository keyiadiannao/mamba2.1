from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import build_batch_summary, load_json, run_navigation_sample
from src.tracing import append_jsonl, make_run_id, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of navigation samples.")
    parser.add_argument(
        "--config",
        default="configs/experiment/navigation_batch_demo.json",
        help="Path to the batch experiment config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / Path(args.config))
    samples_path = ROOT / str(config["samples_path"])
    samples_payload = load_json(samples_path)
    samples = list(samples_payload.get("samples", []))
    if not samples:
        raise ValueError("Batch config requires a non-empty samples list.")

    batch_id = make_run_id(str(config.get("batch_id_prefix", "nav_batch")))
    sample_payloads = []
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id", f"sample_{index:03d}"))
        payload = run_navigation_sample(
            root_dir=ROOT,
            config=config,
            question=str(sample.get("question") or config.get("question") or ""),
            tree_path=str(sample.get("tree_path") or config["tree_path"]),
            reference_answer=str(sample["reference_answer"]) if sample.get("reference_answer") else None,
            run_id_prefix=f"{config.get('run_id_prefix', 'nav_batch')}_{sample_id}",
            sample_id=sample_id,
        )
        sample_payloads.append(payload)

    batch_summary = build_batch_summary(batch_id, sample_payloads)
    batch_summary["config"] = config
    batch_summary["samples_path"] = str(config["samples_path"])

    batch_output_dir = ROOT / str(config.get("batch_output_dir", "outputs/reports/batches"))
    write_json(batch_output_dir / batch_id / "batch_summary.json", batch_summary)
    append_jsonl(ROOT / "outputs" / "reports" / "batch_summary.jsonl", batch_summary)

    print(json.dumps(batch_summary, indent=2, ensure_ascii=False))
    print(f"\nSaved batch summary to: {batch_output_dir / batch_id / 'batch_summary.json'}")
    print(f"Updated batch summary registry: {ROOT / 'outputs' / 'reports' / 'batch_summary.jsonl'}")


if __name__ == "__main__":
    main()
