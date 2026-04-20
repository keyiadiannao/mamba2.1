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
            "last / attn across audit-style labels (touched/never_visit/visit_miss/etc.)."
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


def _leaf_indices_from_node_ids(node_ids: list[str], node_index: dict[str, Any]) -> set[int]:
    leaves: set[int] = set()
    for node_id in node_ids:
        node = node_index.get(node_id)
        if node is None:
            continue
        leaf_idx = node.metadata.get("leaf_index")
        if isinstance(leaf_idx, int):
            leaves.add(int(leaf_idx))
    return leaves


def _derive_labels(sample: dict[str, Any], trace: Any, node_index: dict[str, Any]) -> dict[str, int] | None:
    gold = set(int(x) for x in sample.get("positive_leaf_indices", []) if isinstance(x, int))
    if not gold:
        return None
    visited = set(int(x) for x in (trace.visited_leaf_indices_deduped or []) if isinstance(x, int))
    visited_gold = visited.intersection(gold)

    accepted_node_ids = [str(x) for x in (trace.evidence_node_ids or [])]
    accepted_leafs = _leaf_indices_from_node_ids(accepted_node_ids, node_index)
    accepted_gold = accepted_leafs.intersection(gold)

    context_node_ids = [str(x) for x in (trace.context_node_ids or [])]
    context_leafs = _leaf_indices_from_node_ids(context_node_ids, node_index)
    context_gold = context_leafs.intersection(gold)

    never_visit = 1 if not visited_gold else 0
    visit_miss = 1 if (visited_gold and not visited_gold.issubset(accepted_gold)) else 0
    visit_clean = 1 if (visited_gold and visited_gold.issubset(accepted_gold)) else 0
    any_gold_in_context = 1 if context_gold else 0
    touched_gold = 1 if visited_gold else 0
    return {
        "touched_gold": touched_gold,
        "never_visit": never_visit,
        "visit_miss": visit_miss,
        "visit_clean": visit_clean,
        "any_gold_in_context": any_gold_in_context,
    }


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

    label_keys = [
        "touched_gold",
        "never_visit",
        "visit_miss",
        "visit_clean",
        "any_gold_in_context",
    ]
    labels_by_target: dict[str, list[int]] = {k: [] for k in label_keys}
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

        labels = _derive_labels(sample, trace, node_index)
        if labels is None:
            dropped += 1
            dropped_reasons["no_gold_annotation"] = dropped_reasons.get("no_gold_annotation", 0) + 1
            continue

        for k in label_keys:
            labels_by_target[k].append(int(labels[k]))
        for k, v in method_score.items():
            score_map[k].append(float(v))

        if idx % 20 == 0:
            print(f"[path-smoke] processed {idx}/{len(samples)} samples", flush=True)

    result: dict[str, Any] = {
        "sample_count": len(labels_by_target["touched_gold"]),
        "dropped_samples": dropped,
        "dropped_reasons": dropped_reasons,
        "targets": {},
        "config_snapshot": {
            "navigator_type": config.get("navigator_type"),
            "navigator_model_name": config.get("navigator_model_name"),
            "routing_mode": config.get("routing_mode"),
            "entity_boost_alpha": config.get("entity_boost_alpha"),
            "force_native": bool(args.force_native),
            "max_samples": int(args.max_samples),
        },
    }

    for target, labels in labels_by_target.items():
        target_payload: dict[str, Any] = {
            "positive_count": int(sum(labels)),
            "negative_count": int(len(labels) - sum(labels)),
            "metrics": {},
        }
        for method, scores in score_map.items():
            pos_scores = [s for s, y in zip(scores, labels) if y == 1]
            neg_scores = [s for s, y in zip(scores, labels) if y == 0]
            auc = _roc_auc(scores, labels)
            auc_inv = (1.0 - auc) if auc is not None else None
            best_auc = None if auc is None else max(auc, 1.0 - auc)
            direction = None
            if auc is not None:
                direction = "higher_score=>label1" if auc >= 0.5 else "lower_score=>label1"
            target_payload["metrics"][method] = {
                "mean_score_pos": (sum(pos_scores) / len(pos_scores)) if pos_scores else None,
                "mean_score_neg": (sum(neg_scores) / len(neg_scores)) if neg_scores else None,
                "delta_pos_neg": (
                    (sum(pos_scores) / len(pos_scores)) - (sum(neg_scores) / len(neg_scores))
                    if pos_scores and neg_scores
                    else None
                ),
                "roc_auc": auc,
                "roc_auc_inverted": auc_inv,
                "best_roc_auc_any_direction": best_auc,
                "preferred_direction": direction,
            }
        result["targets"][target] = target_payload

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
