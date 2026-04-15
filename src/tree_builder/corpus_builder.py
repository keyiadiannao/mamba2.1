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


def _first_non_empty(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


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


def build_doc_leaf_index_map(tree_payload: dict[str, Any]) -> dict[str, list[int]]:
    root_payload = tree_payload.get("root", {})
    if not isinstance(root_payload, dict):
        raise ValueError("Tree payload must include a root object.")

    doc_leaf_index_map: dict[str, list[int]] = {}

    def collect_leaf_indices(node: dict[str, Any]) -> list[int]:
        children = node.get("children", [])
        if not isinstance(children, list) or not children:
            leaf_index = node.get("leaf_index")
            if isinstance(leaf_index, int):
                return [leaf_index]
            metadata = node.get("metadata", {})
            metadata_leaf_index = metadata.get("leaf_index") if isinstance(metadata, dict) else None
            return [metadata_leaf_index] if isinstance(metadata_leaf_index, int) else []

        leaf_indices: list[int] = []
        for child in children:
            if isinstance(child, dict):
                leaf_indices.extend(collect_leaf_indices(child))
        return leaf_indices

    def walk(node: dict[str, Any]) -> None:
        children = node.get("children", [])
        metadata = node.get("metadata", {})
        doc_id = metadata.get("doc_id") if isinstance(metadata, dict) else None
        if isinstance(doc_id, str) and doc_id and isinstance(children, list) and children:
            doc_leaf_index_map[doc_id] = sorted(set(collect_leaf_indices(node)))

        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    walk(child)

    walk(root_payload)
    return doc_leaf_index_map


def build_navigation_samples_from_qa(
    qa_records: list[dict[str, Any]],
    *,
    tree_payload: dict[str, Any],
    tree_path: str,
) -> dict[str, Any]:
    doc_leaf_index_map = build_doc_leaf_index_map(tree_payload)
    samples: list[dict[str, Any]] = []

    for sample_index, record in enumerate(qa_records):
        question = str(record.get("question") or "").strip()
        if not question:
            raise ValueError(f"QA record {sample_index} is missing a question.")

        sample_id = str(record.get("sample_id") or f"sample_{sample_index:03d}")
        reference_answer = str(record.get("reference_answer") or "").strip()
        positive_doc_ids = record.get("positive_doc_ids", [])
        if not isinstance(positive_doc_ids, list):
            raise ValueError(f"QA record {sample_id} must provide positive_doc_ids as a list.")

        positive_leaf_indices: list[int] = []
        for doc_id in positive_doc_ids:
            if not isinstance(doc_id, str):
                continue
            positive_leaf_indices.extend(doc_leaf_index_map.get(doc_id, []))

        sample_payload: dict[str, Any] = {
            "sample_id": sample_id,
            "question": question,
            "tree_path": tree_path,
            "positive_leaf_indices": sorted(set(positive_leaf_indices)),
        }
        if reference_answer:
            sample_payload["reference_answer"] = reference_answer
        samples.append(sample_payload)

    return {"samples": samples}


def build_corpus_and_qa_from_wiki_longdoc_samples(
    sample_records: list[dict[str, Any]],
    *,
    source_name: str = "wiki_longdoc_subset",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus_by_doc_id: dict[str, dict[str, Any]] = {}
    qa_records: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(sample_records):
        question = _first_non_empty(sample.get("question"))
        if question is None:
            raise ValueError(f"Sample {sample_index} is missing a question.")

        sample_id = _first_non_empty(sample.get("sample_id")) or f"sample_{sample_index:03d}"
        reference_answer = _first_non_empty(sample.get("reference_answer"), sample.get("answer"))
        pages = sample.get("pages", [])
        if not isinstance(pages, list) or not pages:
            raise ValueError(f"Sample {sample_id} must provide a non-empty pages list.")

        page_to_section_ids: dict[str, list[str]] = {}
        for page_index, page in enumerate(pages):
            if not isinstance(page, dict):
                continue
            page_title = _first_non_empty(page.get("title")) or f"page_{page_index:03d}"
            page_id = _first_non_empty(page.get("page_id")) or _slugify(page_title)
            lead_text = _first_non_empty(page.get("lead_text"), page.get("summary"), page_title) or page_title
            sections = page.get("sections", [])
            if not isinstance(sections, list):
                continue

            section_ids: list[str] = []
            for section_index, section in enumerate(sections):
                if not isinstance(section, dict):
                    continue
                heading = _first_non_empty(section.get("heading"), section.get("title")) or f"Section {section_index + 1}"
                raw_section_id = _first_non_empty(section.get("section_id"))
                section_id = raw_section_id or f"{page_id}__{_slugify(heading)}_{section_index:03d}"

                paragraphs = section.get("paragraphs", [])
                if not isinstance(paragraphs, list):
                    continue
                paragraph_texts = [paragraph.strip() for paragraph in paragraphs if isinstance(paragraph, str) and paragraph.strip()]
                if not paragraph_texts:
                    continue

                record = {
                    "doc_id": section_id,
                    "group_id": page_id,
                    "group_title": page_title,
                    "group_text": lead_text,
                    "title": f"{page_title} / {heading}",
                    "summary": heading,
                    "text": "\n\n".join(paragraph_texts),
                    "source": source_name,
                }

                existing = corpus_by_doc_id.get(section_id)
                if existing is not None and existing != record:
                    raise ValueError(f"Conflicting section payload for doc_id: {section_id}")
                corpus_by_doc_id[section_id] = record
                section_ids.append(section_id)

            page_to_section_ids[page_id] = section_ids

        positive_doc_ids_raw = sample.get("supporting_section_ids")
        positive_doc_ids: list[str] = []
        if isinstance(positive_doc_ids_raw, list) and positive_doc_ids_raw:
            positive_doc_ids = [doc_id for doc_id in positive_doc_ids_raw if isinstance(doc_id, str) and doc_id]
        else:
            supporting_page_ids = sample.get("supporting_page_ids", [])
            if isinstance(supporting_page_ids, list):
                for page_id in supporting_page_ids:
                    if isinstance(page_id, str) and page_id:
                        positive_doc_ids.extend(page_to_section_ids.get(page_id, []))

        qa_record: dict[str, Any] = {
            "sample_id": sample_id,
            "question": question,
            "positive_doc_ids": sorted(set(positive_doc_ids)),
        }
        if reference_answer is not None:
            qa_record["reference_answer"] = reference_answer
        qa_records.append(qa_record)

    corpus_records = list(corpus_by_doc_id.values())
    return corpus_records, qa_records


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
    grouped_children: dict[str, dict[str, Any]] = {}
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

        branch_payload = {
            "node_id": f"branch_{doc_id}",
            "text": str(record.get("summary") or title),
            "metadata": {
                "doc_id": doc_id,
                "title": title,
                "source": record.get("source"),
            },
            "children": leaf_nodes,
        }

        group_id = record.get("group_id")
        if isinstance(group_id, str) and group_id:
            group_title = str(record.get("group_title") or group_id)
            group_text = str(record.get("group_text") or record.get("group_summary") or group_title)
            group_node = grouped_children.get(group_id)
            if group_node is None:
                group_node = {
                    "node_id": f"group_{group_id}",
                    "text": group_text,
                    "metadata": {
                        "group_id": group_id,
                        "title": group_title,
                        "source": record.get("source"),
                    },
                    "children": [],
                }
                grouped_children[group_id] = group_node
                root_children.append(group_node)
            group_node["children"].append(branch_payload)
        else:
            root_children.append(branch_payload)

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
