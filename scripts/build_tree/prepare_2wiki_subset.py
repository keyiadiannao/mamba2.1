from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tree_builder import build_wiki_longdoc_samples_from_2wiki, load_corpus_jsonl


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 2WikiMultiHopQA samples into normalized wiki-longdoc jsonl format."
    )
    parser.add_argument("--input", required=True, help="Input 2Wiki jsonl path.")
    parser.add_argument("--output", required=True, help="Output normalized wiki-longdoc jsonl path.")
    parser.add_argument(
        "--sentences-per-section",
        type=int,
        default=5,
        help="Number of context sentences grouped into one synthetic section.",
    )
    parser.add_argument(
        "--lead-sentences",
        type=int,
        default=2,
        help="Number of leading sentences used as page-level summary text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_records = load_corpus_jsonl(ROOT / Path(args.input))
    normalized_records = build_wiki_longdoc_samples_from_2wiki(
        sample_records,
        sentences_per_section=args.sentences_per_section,
        lead_sentences=args.lead_sentences,
    )

    output_path = ROOT / Path(args.output)
    _write_jsonl(output_path, normalized_records)

    print(
        json.dumps(
            {
                "input_sample_count": len(sample_records),
                "output_path": str(output_path),
                "normalized_sample_count": len(normalized_records),
                "sentences_per_section": args.sentences_per_section,
                "lead_sentences": args.lead_sentences,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
