from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tree_builder import (
    build_navigation_samples_from_qa,
    build_tree_payload_from_corpus,
    load_corpus_jsonl,
)


def _to_repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(f"Path must stay inside repository root: {path}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tree payload and batch manifest from corpus and QA jsonl files."
    )
    parser.add_argument("--corpus-input", required=True, help="Path to corpus jsonl.")
    parser.add_argument("--qa-input", required=True, help="Path to QA jsonl.")
    parser.add_argument("--tree-output", required=True, help="Output tree payload JSON path.")
    parser.add_argument("--batch-output", required=True, help="Output batch manifest JSON path.")
    parser.add_argument(
        "--max-chars-per-leaf",
        type=int,
        default=400,
        help="Maximum approximate characters per leaf chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_records = load_corpus_jsonl(ROOT / Path(args.corpus_input))
    qa_records = load_corpus_jsonl(ROOT / Path(args.qa_input))

    tree_output_path = ROOT / Path(args.tree_output)
    batch_output_path = ROOT / Path(args.batch_output)
    tree_output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_output_path.parent.mkdir(parents=True, exist_ok=True)

    tree_payload = build_tree_payload_from_corpus(
        corpus_records,
        max_chars_per_leaf=args.max_chars_per_leaf,
    )
    tree_output_path.write_text(json.dumps(tree_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    tree_path_relative = _to_repo_relative(tree_output_path)
    batch_payload = build_navigation_samples_from_qa(
        qa_records,
        tree_payload=tree_payload,
        tree_path=tree_path_relative,
    )
    batch_output_path.write_text(json.dumps(batch_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "tree_output_path": str(tree_output_path),
                "batch_output_path": str(batch_output_path),
                "document_count": len(corpus_records),
                "sample_count": len(batch_payload["samples"]),
                "tree_sha256": tree_payload.get("tree_sha256"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
