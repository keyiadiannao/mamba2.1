from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight learned router head.")
    parser.add_argument(
        "--input",
        default="outputs/reports/router_training_data.jsonl",
        help="Input router-training jsonl path.",
    )
    parser.add_argument(
        "--output",
        default="configs/router/learned_router_demo.json",
        help="Output checkpoint json path.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="(sigmoid only) Multiply gradients for label==1 rows. Default: min(neg/pos, 200) when pos>0, else 1.",
    )
    parser.add_argument(
        "--loss",
        choices=("listwise_softmax", "sigmoid"),
        default="listwise_softmax",
        help="listwise_softmax: CE with uniform target over positive children per (sample_id, parent_node_id). "
        "sigmoid: legacy independent logistic regression on each row.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _stable_softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    if s <= 0.0:
        n = len(logits)
        return [1.0 / n] * n
    return [e / s for e in exps]


def _group_row_indices(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (str(row.get("sample_id", "")), str(row.get("parent_node_id", "")))
        groups[key].append(index)
    ordered: dict[tuple[str, str], list[int]] = {}
    for key, indices in groups.items():
        ordered[key] = sorted(indices, key=lambda i: str(rows[i].get("child_node_id", "")))
    return ordered


def _train_sigmoid(
    feature_matrix: list[list[float]],
    labels: list[float],
    *,
    epochs: int,
    lr: float,
    pos_weight: float,
) -> tuple[list[float], float]:
    weights = [0.0 for _ in range(len(feature_matrix[0]))]
    bias = 0.0
    n_rows = len(feature_matrix)
    for _ in range(epochs):
        grad_weights = [0.0 for _ in weights]
        grad_bias = 0.0
        for features, label in zip(feature_matrix, labels):
            logit = bias + sum(weight * feature for weight, feature in zip(weights, features))
            prediction = sigmoid(logit)
            error = prediction - label
            sample_w = pos_weight if label >= 0.5 else 1.0
            for index, feature in enumerate(features):
                grad_weights[index] += error * feature * sample_w
            grad_bias += error * sample_w
        scale = 1.0 / max(n_rows, 1)
        for index in range(len(weights)):
            weights[index] -= lr * grad_weights[index] * scale
        bias -= lr * grad_bias * scale
    return weights, bias


def _train_listwise_softmax(
    rows: list[dict[str, object]],
    feature_matrix: list[list[float]],
    labels: list[float],
    *,
    epochs: int,
    lr: float,
) -> tuple[list[float], float, dict[str, int]]:
    n_features = len(feature_matrix[0])
    weights = [0.0 for _ in range(n_features)]
    bias = 0.0
    groups = _group_row_indices(rows)
    n_rows_used = 0
    skipped_all_negative = 0
    for indices in groups.values():
        pos_sum = sum(labels[i] for i in indices)
        if pos_sum <= 0.0:
            skipped_all_negative += 1
            continue
        n_rows_used += len(indices)

    if n_rows_used == 0:
        raise ValueError("listwise_softmax: no group with at least one positive label.")

    for _ in range(epochs):
        grad_weights = [0.0 for _ in weights]
        grad_bias = 0.0
        for indices in groups.values():
            pos_sum = sum(labels[i] for i in indices)
            if pos_sum <= 0.0:
                continue
            xs = [feature_matrix[i] for i in indices]
            target = [labels[i] / pos_sum for i in indices]
            logits = [bias + sum(w * f for w, f in zip(weights, x)) for x in xs]
            probs = _stable_softmax(logits)
            for x, p, y in zip(xs, probs, target):
                err = p - y
                for j, feature in enumerate(x):
                    grad_weights[j] += err * feature
                grad_bias += err
        scale = 1.0 / max(n_rows_used, 1)
        for index in range(len(weights)):
            weights[index] -= lr * grad_weights[index] * scale
        bias -= lr * grad_bias * scale

    meta = {
        "n_groups": len(groups),
        "skipped_all_negative_groups": skipped_all_negative,
        "n_rows_used_in_listwise": n_rows_used,
    }
    return weights, bias, meta


def main() -> None:
    args = parse_args()
    rows = load_rows(ROOT / Path(args.input))
    if not rows:
        raise ValueError("No training rows found.")

    feature_names = sorted(rows[0]["features"].keys())
    for row in rows[1:]:
        if sorted(row["features"].keys()) != feature_names:
            raise ValueError("All rows must share the same feature keys as the first row.")

    feature_matrix = [
        [float(row["features"][name]) for name in feature_names]
        for row in rows
    ]
    labels = [float(row["label"]) for row in rows]
    n_pos = int(sum(labels))
    n_neg = len(labels) - n_pos

    if args.loss == "sigmoid":
        if args.pos_weight is not None:
            pos_weight = float(args.pos_weight)
        elif n_pos > 0:
            pos_weight = min(float(n_neg) / float(n_pos), 200.0)
        else:
            pos_weight = 1.0
        weights, bias = _train_sigmoid(
            feature_matrix, labels, epochs=args.epochs, lr=args.lr, pos_weight=pos_weight
        )
        listwise_meta: dict[str, int] | None = None
    else:
        weights, bias, listwise_meta = _train_listwise_softmax(
            rows, feature_matrix, labels, epochs=args.epochs, lr=args.lr
        )

    checkpoint: dict[str, object] = {
        "feature_names": feature_names,
        "weights": weights,
        "bias": bias,
        "epochs": args.epochs,
        "lr": args.lr,
        "row_count": len(rows),
        "loss": args.loss,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    if args.loss == "sigmoid":
        checkpoint["pos_weight"] = pos_weight
    else:
        checkpoint["pos_weight"] = None
        checkpoint.update(listwise_meta or {})

    output_path = ROOT / Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"output_path": str(output_path), "row_count": len(rows), "loss": args.loss}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
