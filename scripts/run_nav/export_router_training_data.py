from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.navigator import NavigatorState
from src.pipeline import load_json
from src.router import extract_router_features
from src.tree_builder import load_tree_from_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export learned-router training data.")
    parser.add_argument(
        "--samples",
        default="data/processed/demo_navigation_batch.json",
        help="Path to the batch sample manifest.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/router_training_data.jsonl",
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--root-only",
        action="store_true",
        help="Only export root->child rows (target root routing first).",
    )
    return parser.parse_args()


def subtree_leaf_indices(node) -> set[int]:
    if node.is_leaf:
        leaf_index = node.metadata.get("leaf_index")
        return {int(leaf_index)} if isinstance(leaf_index, int) else set()

    indices: set[int] = set()
    for child in node.children:
        indices.update(subtree_leaf_indices(child))
    return indices


def walk_parent_child_rows(
    question: str,
    node,
    positive_leaf_indices: set[int],
    *,
    depth: int = 0,
    root_only: bool = False,
):
    state = NavigatorState(relevance_score=float(depth))
    for child in node.children:
        child_leaf_indices = subtree_leaf_indices(child)
        features = extract_router_features(question, child, state)
        label = int(bool(child_leaf_indices.intersection(positive_leaf_indices)))
        yield {
            "parent_node_id": node.node_id,
            "child_node_id": child.node_id,
            "depth": depth,
            "label": label,
            "positive_leaf_indices": sorted(positive_leaf_indices),
            "child_leaf_indices": sorted(child_leaf_indices),
            "features": features,
        }
        if not root_only and not child.is_leaf:
            yield from walk_parent_child_rows(
                question,
                child,
                positive_leaf_indices,
                depth=depth + 1,
                root_only=root_only,
            )


def main() -> None:
    args = parse_args()
    samples_payload = load_json(ROOT / Path(args.samples))
    rows = []
    for sample in samples_payload.get("samples", []):
        tree = load_tree_from_json(ROOT / Path(sample["tree_path"]))
        question = str(sample["question"])
        positive_leaf_indices = {int(index) for index in sample.get("positive_leaf_indices", [])}
        for row in walk_parent_child_rows(
            question,
            tree.root,
            positive_leaf_indices,
            root_only=bool(args.root_only),
        ):
            row["sample_id"] = sample["sample_id"]
            row["question"] = question
            rows.append(row)

    output_path = ROOT / Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"output_path": str(output_path), "row_count": len(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
