from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import normalize_reference_for_scoring
from src.pipeline import build_batch_summary, build_controller, load_json, run_navigation_sample
from src.tracing import append_jsonl, make_run_id, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of navigation samples.")
    parser.add_argument(
        "--config",
        default="configs/experiment/navigation_batch_demo.json",
        help="Path to the batch experiment config JSON.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use only the first N samples from the manifest (smoke / alpha sweep). Full manifest is unchanged.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(ROOT / Path(args.config))
    samples_path = ROOT / str(config["samples_path"])
    samples_payload = load_json(samples_path)
    samples = list(samples_payload.get("samples", []))
    manifest_sample_count = len(samples)
    if not samples:
        raise ValueError("Batch config requires a non-empty samples list.")

    max_samples = args.max_samples
    if max_samples is not None:
        n = int(max_samples)
        if n <= 0:
            raise ValueError("--max-samples must be positive.")
        samples = samples[:n]

    batch_id = make_run_id(str(config.get("batch_id_prefix", "nav_batch")))
    controller = build_controller(config)
    sample_payloads = []
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id", f"sample_{index:03d}"))
        payload = run_navigation_sample(
            root_dir=ROOT,
            config=config,
            question=str(sample.get("question") or config.get("question") or ""),
            tree_path=str(sample.get("tree_path") or config["tree_path"]),
            reference_answer=normalize_reference_for_scoring(sample.get("reference_answer")),
            run_id_prefix=f"{config.get('run_id_prefix', 'nav_batch')}_{sample_id}",
            sample_id=sample_id,
            batch_id=batch_id,
            leaf_indices_required=list(sample.get("positive_leaf_indices", [])),
            controller=controller,
        )
        sample_payloads.append(payload)

    batch_summary = build_batch_summary(batch_id, sample_payloads)
    batch_summary["config"] = config
    batch_summary["samples_path"] = str(config["samples_path"])
    batch_summary["manifest_sample_count"] = manifest_sample_count
    batch_summary["max_samples"] = max_samples

    batch_output_dir = ROOT / str(config.get("batch_output_dir", "outputs/reports/batches"))
    write_json(batch_output_dir / batch_id / "batch_summary.json", batch_summary)
    append_jsonl(ROOT / "outputs" / "reports" / "batch_summary.jsonl", batch_summary)

    print(json.dumps(batch_summary, indent=2, ensure_ascii=False))
    print(f"\nSaved batch summary to: {batch_output_dir / batch_id / 'batch_summary.json'}")
    print(f"Updated batch summary registry: {ROOT / 'outputs' / 'reports' / 'batch_summary.jsonl'}")
    # Machine-parseable line for ad-hoc shell (sed/grep); no wrapper script in-repo.
    print(f"__SSGS_BATCH_ID__={batch_id}", flush=True)


if __name__ == "__main__":
    main()
