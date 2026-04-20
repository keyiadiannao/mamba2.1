from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import build_controller, load_json
from src.tree_builder import load_tree_from_payload, load_tree_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test for path representation usefulness: compare mamba_seq / mean / "
            "last / attn on touched-gold discrimination."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/experiment/navigation_batch_real_corpus_p0_visit_rule_entity_boost_a030.example.json",
        help="Batch config JSON.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Evaluate first N samples from manifest.",
    )
    parser.add_argument(
        "--force-native",
        action="store_true",
        help="Override config to local native navigator (mamba2_native + smoke_block).",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output path; defaults to outputs/reports/path_state_smoke_<timestamp>.json",
    )
    return parser.parse_args()


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    n1 = _norm(left)
    n2 = _norm(right)
    if n1 <= 0.0 or n2 <= 0.0:
        return 0.0
    return _dot(left, right) / (n1 * n2)


def _vector_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = min(len(v) for v in vectors)
    if dim <= 0:
        return []
    sums = [0.0] * dim
    for vec in vectors:
        for i in range(dim):
            sums[i] += float(vec[i])
    n = float(len(vectors))
    return [v / n for v in sums]


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    vmax = max(values)
    exps = [math.exp(v - vmax) for v in values]
    s = sum(exps)
    if s <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [e / s for e in exps]


def _weighted_sum(vectors: list[list[float]], weights: list[float]) -> list[float]:
    if not vectors:
        return []
    dim = min(len(v) for v in vectors)
    out = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            out[i] += float(w) * float(vec[i])
    return out


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _embed_question(navigator: Any, question: str) -> list[float]:
    if hasattr(navigator, "_encode_query"):
        return [float(x) for x in navigator._encode_query(question)]  # type: ignore[attr-defined]
    if hasattr(navigator, "_encode_text"):
        return [float(x) for x in navigator._encode_text(question, None)]  # type: ignore[attr-defined]
    raise RuntimeError("Navigator has no query/text encoding hook for smoke test.")


def _embed_node_independent(navigator: Any, text: str) -> list[float]:
    if hasattr(navigator, "_encode_text"):
        return [float(x) for x in navigator._encode_text(text, None)]  # type: ignore[attr-defined]
    if hasattr(navigator, "_encode_node_text"):
        return [float(x) for x in navigator._encode_node_text(text, None)]  # type: ignore[attr-defined]
    raise RuntimeError("Navigator has no node encoding hook for smoke test.")


def _embed_sequence_mamba_style(navigator: Any, texts: list[str]) -> list[float]:
    summary: list[float] | None = None
    if hasattr(navigator, "_encode_text"):
        for t in texts:
            summary = [float(x) for x in navigator._encode_text(t, summary)]  # type: ignore[attr-defined]
        return summary or []
    if hasattr(navigator, "_encode_node_text"):
        for t in texts:
            summary = [float(x) for x in navigator._encode_node_text(t, summary)]  # type: ignore[attr-defined]
        return summary or []
    raise RuntimeError("Navigator has no sequence encoding hook for smoke test.")


def _method_scores(question_vec: list[float], node_vecs: list[list[float]], navigator: Any, node_texts: list[str]) -> dict[str, float]:
    if not node_vecs:
        return {"mamba_seq": 0.0, "mean": 0.0, "last": 0.0, "attn": 0.0}
    mamba_seq_vec = _embed_sequence_mamba_style(navigator, node_texts)
    mean_vec = _vector_mean(node_vecs)
    last_vec = node_vecs[-1]
    attn_logits = [_cosine(question_vec, v) for v in node_vecs]
    attn_weights = _softmax(attn_logits)
    attn_vec = _weighted_sum(node_vecs, attn_weights)
    return {
        "mamba_seq": _cosine(question_vec, mamba_seq_vec),
        "mean": _cosine(question_vec, mean_vec),
        "last": _cosine(question_vec, last_vec),
        "attn": _cosine(question_vec, attn_vec),
    }


def _prepare_config(config: dict[str, Any], force_native: bool) -> dict[str, Any]:
    out = dict(config)
    if not force_native:
        return out
    out["navigator_type"] = "mamba2_native"
    out["navigator_load_strategy"] = "smoke_block"
    out["navigator_model_name"] = str(config.get("navigator_model_name", "mamba2"))
    out["navigator_dependency_module"] = "mamba2_native"
    out["navigator_device"] = "cuda"
    out["navigator_dtype"] = "float16"
    out.pop("navigator_pretrained_checkpoint", None)
    out.pop("navigator_tokenizer_name", None)
    out.pop("navigator_path_recursive_prompt", None)
    return out


def main() -> None:
    args = parse_args()
    config = _prepare_config(load_json(ROOT / args.config), bool(args.force_native))

    samples_payload = load_json(ROOT / str(config["samples_path"]))
    samples = list(samples_payload.get("samples", []))
    if not samples:
        raise ValueError("No samples found in manifest.")
    if args.max_samples and args.max_samples > 0:
        samples = samples[: int(args.max_samples)]

    controller = build_controller(config)
    navigator = controller.navigator

    labels: list[int] = []
    score_map: dict[str, list[float]] = {"mamba_seq": [], "mean": [], "last": [], "attn": []}
    dropped = 0
    dropped_reasons: dict[str, int] = {}

    for idx, sample in enumerate(samples, start=1):
        tree_path = ROOT / str(sample.get("tree_path") or config["tree_path"])
        tree = load_tree_from_payload(load_tree_payload(tree_path))
        trace = controller.run(str(sample.get("question", "")), tree)
        node_index = tree.build_node_index()
        visited_ids = list(trace.visited_node_ids or [])
        if not visited_ids:
            dropped += 1
            reason = str(trace.context_build_error or trace.failure_attribution or "no_visited_nodes")
            dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
            continue

        question = str(sample.get("question", ""))
        question_vec = _embed_question(navigator, question)
        node_texts = [(node_index[nid].text if nid in node_index else "") for nid in visited_ids]
        node_vecs = [_embed_node_independent(navigator, txt) for txt in node_texts]
        method_score = _method_scores(question_vec, node_vecs, navigator, node_texts)

        gold = set(int(x) for x in sample.get("positive_leaf_indices", []) if isinstance(x, int))
        visited_leaves = set(int(x) for x in (trace.visited_leaf_indices_deduped or []) if isinstance(x, int))
        y = 1 if gold and (gold.intersection(visited_leaves)) else 0

        labels.append(y)
        for k, v in method_score.items():
            score_map[k].append(float(v))

        if idx % 20 == 0:
            print(f"[path-smoke] processed {idx}/{len(samples)} samples", flush=True)

    result: dict[str, Any] = {
        "sample_count": len(labels),
        "dropped_samples": dropped,
        "dropped_reasons": dropped_reasons,
        "positive_count": int(sum(labels)),
        "negative_count": int(len(labels) - sum(labels)),
        "metrics": {},
        "config_snapshot": {
            "navigator_type": config.get("navigator_type"),
            "navigator_model_name": config.get("navigator_model_name"),
            "routing_mode": config.get("routing_mode"),
            "entity_boost_alpha": config.get("entity_boost_alpha"),
            "force_native": bool(args.force_native),
            "max_samples": int(args.max_samples),
        },
    }

    for method, scores in score_map.items():
        pos_scores = [s for s, y in zip(scores, labels) if y == 1]
        neg_scores = [s for s, y in zip(scores, labels) if y == 0]
        result["metrics"][method] = {
            "mean_score_pos": (sum(pos_scores) / len(pos_scores)) if pos_scores else None,
            "mean_score_neg": (sum(neg_scores) / len(neg_scores)) if neg_scores else None,
            "delta_pos_neg": (
                (sum(pos_scores) / len(pos_scores)) - (sum(neg_scores) / len(neg_scores))
                if pos_scores and neg_scores
                else None
            ),
            "roc_auc": _roc_auc(scores, labels),
        }

    out_path = (
        ROOT / args.out_json
        if args.out_json
        else ROOT / "outputs" / "reports" / f"path_state_smoke_{Path(args.config).stem}_{len(labels)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved path-state smoke report to: {out_path}")


if __name__ == "__main__":
    main()
