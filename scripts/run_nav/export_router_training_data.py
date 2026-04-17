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
from src.tree_builder import TreeNode, load_tree_from_json


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
    parser.add_argument(
        "--max-root-children",
        type=int,
        default=128,
        help="When --root-only: cap root fan-out rows per sample (keep all positives, then top negatives by heuristic).",
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


def _root_child_training_rows(
    question: str,
    root: TreeNode,
    positive_leaf_indices: set[int],
    *,
    max_root_children: int,
) -> list[dict[str, object]]:
    """Export capped root->child rows for huge fan-out corpora.

    Real corpus trees can have thousands of direct children under ``root``. Exporting all
    rows yields an unusable class imbalance (near-zero positive rate). We therefore:
    - keep **all** positive-labeled root children,
    - fill remaining budget with highest-scoring negatives under the same lexical+cosine
      heuristic used by ``RuleRouter`` (lexical first, cosine tie-break).
    """
    state = NavigatorState(relevance_score=0.0)
    scored: list[tuple[int, float, float, TreeNode, dict[str, float]]] = []
    for child in root.children:
        child_leaf_indices = subtree_leaf_indices(child)
        feats = extract_router_features(question, child, state)
        label = int(bool(child_leaf_indices.intersection(positive_leaf_indices)))
        lex = float(feats["lexical_overlap"])
        cos = float(feats["cosine_probe"])
        scored.append((label, lex, cos, child, feats))

    positives = [row for row in scored if row[0] == 1]
    negatives = [row for row in scored if row[0] == 0]
    positives.sort(key=lambda t: (t[1], t[2], t[3].node_id), reverse=True)
    negatives.sort(key=lambda t: (t[1], t[2], t[3].node_id), reverse=True)

    cap = max(1, int(max_root_children))
    if len(positives) >= cap:
        chosen = positives[:cap]
    else:
        remaining = cap - len(positives)
        chosen = positives + negatives[: max(0, remaining)]

    out: list[dict[str, object]] = []
    for label, _lex, _cos, child, features in chosen:
        child_leaf_indices = subtree_leaf_indices(child)
        out.append(
            {
                "parent_node_id": root.node_id,
                "child_node_id": child.node_id,
                "depth": 0,
                "label": label,
                "positive_leaf_indices": sorted(positive_leaf_indices),
                "child_leaf_indices": sorted(child_leaf_indices),
                "features": features,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    samples_payload = load_json(ROOT / Path(args.samples))
    rows = []
    for sample in samples_payload.get("samples", []):
        tree = load_tree_from_json(ROOT / Path(sample["tree_path"]))
        question = str(sample["question"])
        positive_leaf_indices = {int(index) for index in sample.get("positive_leaf_indices", [])}
        if args.root_only:
            sample_rows = _root_child_training_rows(
                question,
                tree.root,
                positive_leaf_indices,
                max_root_children=int(args.max_root_children),
            )
            for row in sample_rows:
                row["sample_id"] = sample["sample_id"]
                row["question"] = question
                rows.append(row)
            continue
        for row in walk_parent_child_rows(
            question,
            tree.root,
            positive_leaf_indices,
            root_only=False,
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
