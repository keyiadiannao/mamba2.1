from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TreeNode:
    node_id: str
    text: str
    children: list["TreeNode"] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class DocumentTree:
    root: TreeNode

    def walk_depth_first(self) -> Iterable[TreeNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))
