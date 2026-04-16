"""Gate before bumping default context_select_k: demo batch, mock gen, no GPU."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.pipeline import build_batch_summary, build_controller, load_json, run_navigation_sample
from src.tracing import make_run_id


def _run_demo_batch(config_rel: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    config = load_json(root / config_rel)
    samples_path = root / str(config["samples_path"])
    samples_payload = load_json(samples_path)
    samples = list(samples_payload.get("samples", []))
    batch_id = make_run_id(str(config.get("batch_id_prefix", "demo_smoke")))
    controller = build_controller(config)
    payloads = []
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id", f"sample_{index:03d}"))
        payload = run_navigation_sample(
            root_dir=root,
            config=config,
            question=str(sample.get("question") or config.get("question") or ""),
            tree_path=str(sample.get("tree_path") or config["tree_path"]),
            reference_answer=str(sample["reference_answer"]) if sample.get("reference_answer") else None,
            run_id_prefix=f"{config.get('run_id_prefix', 'demo')}_{sample_id}",
            sample_id=sample_id,
            batch_id=batch_id,
            leaf_indices_required=list(sample.get("positive_leaf_indices", [])),
            controller=controller,
        )
        payloads.append(payload)
    return build_batch_summary(batch_id, payloads), payloads


class DemoCtxselKSmokeBatchTests(unittest.TestCase):
    def test_overlap_k3_and_k4_demo_batches_complete_cleanly(self) -> None:
        for rel in (
            "configs/experiment/end_to_end_batch_demo_smoke_ctxsel_overlap_k3.json",
            "configs/experiment/end_to_end_batch_demo_smoke_ctxsel_overlap_k4.json",
        ):
            summary, payloads = _run_demo_batch(rel)
            self.assertEqual(summary["sample_count"], 3, msg=rel)
            self.assertEqual(summary["nav_success_count"], 3, msg=rel)
            for payload in payloads:
                trace = payload["trace"]
                self.assertIsNone(
                    trace.get("generation_error"),
                    msg=f"{rel} sample={payload.get('sample_id')}",
                )


if __name__ == "__main__":
    unittest.main()
