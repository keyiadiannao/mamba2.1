from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TreeNode:
    node_id: str
    text: str
    children: list["TreeNode"] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class DocumentTree:
    root: TreeNode

    def build_node_index(self) -> dict[str, TreeNode]:
        """Map node_id → TreeNode for O(1) lookup when building path-conditioned prompts."""
        return {node.node_id: node for node in self.walk_depth_first()}

    def walk_depth_first(self) -> Iterable[TreeNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))
