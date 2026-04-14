from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .simple_tree import DocumentTree, TreeNode


def _build_node(node_payload: dict[str, Any]) -> TreeNode:
    if "node_id" not in node_payload or "text" not in node_payload:
        raise ValueError("Each node payload must include 'node_id' and 'text'.")

    children_payload = node_payload.get("children", [])
    metadata = dict(node_payload.get("metadata", {}))
    if "leaf_index" in node_payload and "leaf_index" not in metadata:
        metadata["leaf_index"] = node_payload["leaf_index"]

    return TreeNode(
        node_id=str(node_payload["node_id"]),
        text=str(node_payload["text"]),
        children=[_build_node(child) for child in children_payload],
        metadata=metadata,
    )


def load_tree_payload(path: str | Path) -> dict[str, Any]:
    payload_path = Path(path)
    with payload_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Tree payload must be a JSON object.")

    return payload


def load_tree_from_json(path: str | Path) -> DocumentTree:
    payload = load_tree_payload(path)
    root_payload = payload.get("root", payload)
    return DocumentTree(root=_build_node(root_payload))
