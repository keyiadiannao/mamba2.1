from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tree_builder import build_2wiki_subset, load_corpus_jsonl


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a reproducible 2WikiMultiHopQA subset for experiments.")
    parser.add_argument("--input", required=True, help="Input 2Wiki jsonl path.")
    parser.add_argument("--output", required=True, help="Output subset jsonl path.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of samples to keep.")
    parser.add_argument(
        "--min-context-pages",
        type=int,
        default=2,
        help="Require at least this many context pages.",
    )
    parser.add_argument(
        "--min-supporting-facts",
        type=int,
        default=2,
        help="Require at least this many supporting facts.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_records = load_corpus_jsonl(ROOT / Path(args.input))
    subset = build_2wiki_subset(
        sample_records,
        limit=args.limit,
        min_context_pages=args.min_context_pages,
        min_supporting_facts=args.min_supporting_facts,
        seed=args.seed,
    )

    output_path = ROOT / Path(args.output)
    _write_jsonl(output_path, subset)

    print(
        json.dumps(
            {
                "input_sample_count": len(sample_records),
                "output_path": str(output_path),
                "subset_sample_count": len(subset),
                "limit": args.limit,
                "min_context_pages": args.min_context_pages,
                "min_supporting_facts": args.min_supporting_facts,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
