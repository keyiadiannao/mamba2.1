from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build alpha sweep summary with oracle gaps from end_to_end batch summaries."
    )
    parser.add_argument(
        "--input",
        default="outputs/reports/end_to_end_batch_summary.jsonl",
        help="Path to end_to_end_batch_summary jsonl.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/alpha_sweep_summary.json",
        help="Output json path.",
    )
    parser.add_argument(
        "--alpha-batch-id",
        action="append",
        default=None,
        help="Explicit alpha sweep batch id (can repeat). If omitted, auto-discover by prefix.",
    )
    parser.add_argument(
        "--alpha-batch-prefix",
        default="end_to_end_real_corpus_370m_qwen7b_rule_anticollapse_entityalpha_",
        help="Batch id prefix for auto-discovering alpha sweep arms.",
    )
    parser.add_argument(
        "--oracle-batch-id",
        default=None,
        help="Exact oracle batch id. If omitted, auto-select latest by oracle prefix.",
    )
    parser.add_argument(
        "--oracle-batch-prefix",
        default="end_to_end_real_corpus_370m_qwen7b_oracle_500_",
        help="Batch id prefix for auto-discovering oracle batch.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_alpha(batch_id: str, row: dict[str, Any], prefix: str) -> float | None:
    config = row.get("config")
    if isinstance(config, dict) and config.get("entity_boost_alpha") is not None:
        try:
            return float(config["entity_boost_alpha"])
        except (TypeError, ValueError):
            pass
    if not batch_id.startswith(prefix):
        return None
    tail = batch_id[len(prefix) :]
    match = re.match(r"([0-9]+(?:_[0-9]+)?)", tail)
    if not match:
        return None
    return float(match.group(1).replace("_", "."))


def _pick_latest_rows_by_batch_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        batch_id = str(row.get("batch_id") or "")
        if not batch_id:
            continue
        latest[batch_id] = row
    return latest


def _build_arm_summary(row: dict[str, Any], alpha: float) -> dict[str, Any]:
    return {
        "batch_id": row.get("batch_id"),
        "alpha": alpha,
        "sample_count": int(row.get("sample_count") or 0),
        "exact_match_rate": float(row.get("exact_match_rate") or 0.0),
        "avg_answer_f1": float(row.get("avg_answer_f1") or 0.0),
        "avg_rouge_l_f1": float(row.get("avg_rouge_l_f1") or 0.0),
        "nav_success_rate": float(row.get("nav_success_rate") or 0.0),
    }


def main() -> None:
    args = parse_args()
    input_path = ROOT / Path(args.input)
    if (
        not input_path.exists()
        and args.input == "outputs/reports/end_to_end_batch_summary.jsonl"
    ):
        fallback_input = ROOT / "outputs/reports/batch_summary.jsonl"
        if fallback_input.exists():
            input_path = fallback_input
    output_path = ROOT / Path(args.output)
    rows = _load_jsonl(input_path)
    latest_by_batch_id = _pick_latest_rows_by_batch_id(rows)

    alpha_rows: list[tuple[float, dict[str, Any]]] = []
    if args.alpha_batch_id:
        for batch_id in args.alpha_batch_id:
            row = latest_by_batch_id.get(batch_id)
            if row is None:
                continue
            alpha = _extract_alpha(batch_id, row, args.alpha_batch_prefix)
            if alpha is None:
                alpha = float(row.get("config", {}).get("entity_boost_alpha", 0.0))
            alpha_rows.append((alpha, row))
    else:
        best_per_alpha: dict[float, dict[str, Any]] = {}
        for batch_id, row in latest_by_batch_id.items():
            alpha = _extract_alpha(batch_id, row, args.alpha_batch_prefix)
            if alpha is None:
                continue
            existing = best_per_alpha.get(alpha)
            if existing is None or str(row.get("batch_id") or "") > str(existing.get("batch_id") or ""):
                best_per_alpha[alpha] = row
        alpha_rows = sorted((alpha, row) for alpha, row in best_per_alpha.items())

    oracle_row: dict[str, Any] | None = None
    if args.oracle_batch_id:
        oracle_row = latest_by_batch_id.get(args.oracle_batch_id)
    else:
        candidates = [
            row
            for batch_id, row in latest_by_batch_id.items()
            if batch_id.startswith(args.oracle_batch_prefix)
        ]
        if candidates:
            oracle_row = max(candidates, key=lambda item: str(item.get("batch_id") or ""))

    oracle_metrics: dict[str, Any] | None = None
    if oracle_row is not None:
        oracle_metrics = {
            "batch_id": oracle_row.get("batch_id"),
            "sample_count": int(oracle_row.get("sample_count") or 0),
            "exact_match_rate": float(oracle_row.get("exact_match_rate") or 0.0),
            "avg_answer_f1": float(oracle_row.get("avg_answer_f1") or 0.0),
            "avg_rouge_l_f1": float(oracle_row.get("avg_rouge_l_f1") or 0.0),
        }

    arms = [_build_arm_summary(row, alpha) for alpha, row in alpha_rows]
    for arm in arms:
        if oracle_metrics is None:
            arm["oracle_gap"] = None
            continue
        arm["oracle_gap"] = {
            "exact_match_rate_gap": oracle_metrics["exact_match_rate"] - arm["exact_match_rate"],
            "avg_answer_f1_gap": oracle_metrics["avg_answer_f1"] - arm["avg_answer_f1"],
            "avg_rouge_l_f1_gap": oracle_metrics["avg_rouge_l_f1"] - arm["avg_rouge_l_f1"],
        }

    output = {
        "input_path": str(input_path),
        "alpha_batch_prefix": args.alpha_batch_prefix,
        "oracle_batch_prefix": args.oracle_batch_prefix,
        "alpha_batch_ids": [arm["batch_id"] for arm in arms],
        "oracle_batch_id": oracle_metrics["batch_id"] if oracle_metrics else None,
        "oracle_metrics": oracle_metrics,
        "arms": arms,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved alpha sweep summary to: {output_path}")


if __name__ == "__main__":
    main()
