from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "item"


def _chunk_text(text: str, max_chars: int) -> list[str]:
    stripped = " ".join(text.split())
    if not stripped:
        return []
    if len(stripped) <= max_chars:
        return [stripped]

    words = stripped.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        additional = len(word) if not current else len(word) + 1
        if current and current_len + additional > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += additional
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_corpus_jsonl(path: str | Path) -> list[dict[str, Any]]:
    corpus_path = Path(path)
    records: list[dict[str, Any]] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} must be a JSON object.")
            records.append(payload)
    if not records:
        raise ValueError("Corpus jsonl is empty.")
    return records


def build_tree_payload_from_corpus(
    records: list[dict[str, Any]],
    *,
    question: str = "",
    reference_answer: str | None = None,
    root_node_id: str = "root",
    root_text: str = "real corpus navigation root",
    max_chars_per_leaf: int = 400,
) -> dict[str, Any]:
    root_children: list[dict[str, Any]] = []
    leaf_index = 0

    for doc_index, record in enumerate(records):
        title = str(record.get("title") or record.get("doc_id") or f"document_{doc_index:03d}")
        doc_id = str(record.get("doc_id") or _slugify(title))
        text = str(record.get("text") or "").strip()
        if not text:
            continue

        leaf_nodes: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(_chunk_text(text, max_chars=max_chars_per_leaf)):
            leaf_nodes.append(
                {
                    "node_id": f"leaf_{doc_id}_{chunk_index:03d}",
                    "text": chunk,
                    "leaf_index": leaf_index,
                    "metadata": {
                        "doc_id": doc_id,
                        "title": title,
                        "chunk_index": chunk_index,
                    },
                }
            )
            leaf_index += 1

        if not leaf_nodes:
            continue

        root_children.append(
            {
                "node_id": f"branch_{doc_id}",
                "text": str(record.get("summary") or title),
                "metadata": {
                    "doc_id": doc_id,
                    "title": title,
                    "source": record.get("source"),
                },
                "children": leaf_nodes,
            }
        )

    if not root_children:
        raise ValueError("No valid documents with text were found in the corpus.")

    payload: dict[str, Any] = {
        "question": question,
        "root": {
            "node_id": root_node_id,
            "text": root_text,
            "metadata": {"title": "Corpus Root"},
            "children": root_children,
        },
    }
    if reference_answer is not None:
        payload["reference_answer"] = reference_answer
    return payload
