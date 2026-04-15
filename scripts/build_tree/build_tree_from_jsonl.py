from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tree_builder import build_tree_payload_from_corpus, load_corpus_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simple navigation tree payload from a corpus jsonl.")
    parser.add_argument("--input", required=True, help="Path to a corpus jsonl file.")
    parser.add_argument("--output", required=True, help="Output tree payload JSON path.")
    parser.add_argument("--question", default="", help="Optional default question stored in the payload.")
    parser.add_argument("--reference-answer", default=None, help="Optional reference answer stored in the payload.")
    parser.add_argument(
        "--max-chars-per-leaf",
        type=int,
        default=400,
        help="Maximum approximate characters per leaf chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_corpus_jsonl(ROOT / Path(args.input))
    payload = build_tree_payload_from_corpus(
        records,
        question=args.question,
        reference_answer=args.reference_answer,
        max_chars_per_leaf=args.max_chars_per_leaf,
    )

    output_path = ROOT / Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "document_count": len(records),
                "root_child_count": len(payload["root"]["children"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
