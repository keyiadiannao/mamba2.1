from __future__ import annotations

import argparse
import json
import math
import sys
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
        help="Multiply gradients for label==1 rows. Default: min(neg/pos, 200) when pos>0, else 1.",
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


def main() -> None:
    args = parse_args()
    rows = load_rows(ROOT / Path(args.input))
    if not rows:
        raise ValueError("No training rows found.")

    feature_names = sorted(rows[0]["features"].keys())
    feature_matrix = [
        [float(row["features"][name]) for name in feature_names]
        for row in rows
    ]
    labels = [float(row["label"]) for row in rows]

    n_pos = int(sum(labels))
    n_neg = len(labels) - n_pos
    if args.pos_weight is not None:
        pos_weight = float(args.pos_weight)
    elif n_pos > 0:
        pos_weight = min(float(n_neg) / float(n_pos), 200.0)
    else:
        pos_weight = 1.0

    weights = [0.0 for _ in feature_names]
    bias = 0.0

    for _ in range(args.epochs):
        grad_weights = [0.0 for _ in feature_names]
        grad_bias = 0.0
        for features, label in zip(feature_matrix, labels):
            logit = bias + sum(weight * feature for weight, feature in zip(weights, features))
            prediction = sigmoid(logit)
            error = prediction - label
            sample_w = pos_weight if label >= 0.5 else 1.0
            for index, feature in enumerate(features):
                grad_weights[index] += error * feature * sample_w
            grad_bias += error * sample_w

        scale = 1.0 / max(len(feature_matrix), 1)
        for index in range(len(weights)):
            weights[index] -= args.lr * grad_weights[index] * scale
        bias -= args.lr * grad_bias * scale

    checkpoint = {
        "feature_names": feature_names,
        "weights": weights,
        "bias": bias,
        "epochs": args.epochs,
        "lr": args.lr,
        "row_count": len(rows),
        "pos_weight": pos_weight,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }

    output_path = ROOT / Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"output_path": str(output_path), "row_count": len(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
