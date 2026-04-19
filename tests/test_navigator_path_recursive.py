from __future__ import annotations

from src.controller import ControllerConfig, SSGSController
from src.navigator.base import (
    BaseNavigator,
    NavigatorState,
    build_path_recursive_prompt_text,
)
from src.router import BaseRouter, ChildScore, RouteDecision
from src.tree_builder import DocumentTree, TreeNode


def test_build_path_recursive_prompt_text_shape() -> None:
    a = TreeNode("a", "ancestor A text")
    b = TreeNode("b", "ancestor B text")
    cur = TreeNode("c", "current leaf text")
    out = build_path_recursive_prompt_text(
        "What is X?",
        (a, b),
        cur,
        max_chars_segment=100,
        max_chars_question=50,
    )
    assert out.startswith("[Q] ")
    assert "[PATH]" in out
    assert "[NODE]" in out
    assert "ancestor A" in out or "ancestor A text"[:20] in out
    assert "current leaf" in out


class _RecordingPathNav(BaseNavigator):
    """Navigator that records whether ancestors were passed (P1 smoke)."""

    uses_path_recursive_prompt = True

    def __init__(self) -> None:
        self.records: list[tuple[str, int]] = []

    def init_state(self) -> NavigatorState:
        return NavigatorState(hidden_summary=[0.0, 0.0])

    def step(
        self,
        question: str,
        node: TreeNode,
        state: NavigatorState,
        *,
        path_ancestor_nodes=None,
    ) -> NavigatorState:
        self.records.append((node.node_id, len(path_ancestor_nodes or [])))
        nxt = state.clone()
        nxt.path.append(node.node_id)
        nxt.relevance_score = 1.0
        return nxt


class _PassthroughRouter(BaseRouter):
    def rank_children(self, question, parent, children, nav_state):
        return RouteDecision(
            ordered_children=list(children),
            child_scores=[ChildScore(node_id=c.node_id, score=1.0) for c in children],
        )


def test_ssgs_controller_passes_path_ancestors_when_navigator_requests() -> None:
    root = TreeNode("root", "root text", children=[TreeNode("c1", "only child", children=[])])
    tree = DocumentTree(root=root)
    nav = _RecordingPathNav()
    ctrl = SSGSController(
        navigator=nav,
        router=_PassthroughRouter(),
        config=ControllerConfig(max_evidence=1, min_relevance_score=0.0, max_depth=4, max_nodes=16),
    )
    trace = ctrl.run("Q?", tree)
    assert trace.nav_success
    ids = [r[0] for r in nav.records]
    assert ids[0] == "root"
    assert nav.records[0][1] == 0
    assert "c1" in ids
    c1_idx = ids.index("c1")
    assert nav.records[c1_idx][1] == 1


def test_document_tree_build_node_index() -> None:
    leaf = TreeNode("leaf", "L")
    root = TreeNode("root", "R", children=[TreeNode("mid", "M", children=[leaf])])
    idx = DocumentTree(root=root).build_node_index()
    assert set(idx.keys()) == {"root", "mid", "leaf"}
    assert idx["leaf"].text == "L"
