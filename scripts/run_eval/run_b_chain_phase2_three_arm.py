#!/usr/bin/env python3
"""
Run the three Phase-B style end-to-end arms on the real-corpus 500 batch (B chain):

  rule + t1_visited_leaves_ordered + overlap_topk(k=4)
  cosine_probe + t1_visited_leaves_ordered + overlap_topk(k=4)
  oracle_item_leaves + context_select off

Requires a local generator snapshot path (AutoDL / offline HF), e.g.:

  python scripts/run_eval/run_b_chain_phase2_three_arm.py \\
    --generator-hf-model-name /root/autodl-tmp/models/Qwen2.5-7B-Instruct

Writes patched JSON under outputs/reports/tmp_phase2_configs/ then invokes
run_end_to_end_batch.py for each arm in order.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ARMS: list[tuple[str, Path]] = [
    (
        "rule",
        _REPO_ROOT
        / "configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_rule.example.json",
    ),
    (
        "cosine_probe",
        _REPO_ROOT
        / "configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_cosine_probe.example.json",
    ),
    (
        "oracle_item_leaves",
        _REPO_ROOT
        / "configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_oracle_item_leaves.example.json",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--generator-hf-model-name",
        required=True,
        help="Local directory or Hub id for Qwen (use local dir on AutoDL when Hub is unreachable).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write patched configs, do not run batches.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = _REPO_ROOT / "outputs" / "reports" / "tmp_phase2_configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = _REPO_ROOT / "scripts" / "run_eval" / "run_end_to_end_batch.py"
    rel_paths: list[str] = []

    for label, src in _ARMS:
        if not src.is_file():
            print(f"Missing config: {src}", file=sys.stderr)
            return 2
        cfg = json.loads(src.read_text(encoding="utf-8"))
        cfg["generator_hf_model_name"] = str(args.generator_hf_model_name).strip()
        tmp = out_dir / f"phase2_patch_{label}.json"
        tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rel = tmp.relative_to(_REPO_ROOT).as_posix()
        rel_paths.append(rel)
        print(f"Wrote {tmp}")

    if args.dry_run:
        print("Dry-run: skipping batch execution.")
        return 0

    for rel in rel_paths:
        cmd = [sys.executable, str(runner), "--config", rel]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
