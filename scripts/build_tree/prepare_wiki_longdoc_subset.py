from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tree_builder import build_corpus_and_qa_from_wiki_longdoc_samples, load_corpus_jsonl


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert normalized wiki-style long-document QA samples into corpus and QA jsonl files."
    )
    parser.add_argument("--input", required=True, help="Input jsonl with wiki-style long-document QA samples.")
    parser.add_argument("--corpus-output", required=True, help="Output corpus jsonl path.")
    parser.add_argument("--qa-output", required=True, help="Output QA jsonl path.")
    parser.add_argument(
        "--source-name",
        default="wiki_longdoc_subset",
        help="Source tag written into generated corpus records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_records = load_corpus_jsonl(ROOT / Path(args.input))
    corpus_records, qa_records = build_corpus_and_qa_from_wiki_longdoc_samples(
        sample_records,
        source_name=args.source_name,
    )

    corpus_output_path = ROOT / Path(args.corpus_output)
    qa_output_path = ROOT / Path(args.qa_output)
    _write_jsonl(corpus_output_path, corpus_records)
    _write_jsonl(qa_output_path, qa_records)

    print(
        json.dumps(
            {
                "input_sample_count": len(sample_records),
                "corpus_output_path": str(corpus_output_path),
                "qa_output_path": str(qa_output_path),
                "corpus_record_count": len(corpus_records),
                "qa_record_count": len(qa_records),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
