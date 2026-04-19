from __future__ import annotations

import sys
from types import ModuleType


def _install_fake_sentence_transformers() -> None:
    if "sentence_transformers" in sys.modules:
        return

    fake_mod = ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._dim = 4

        def eval(self) -> "FakeSentenceTransformer":
            return self

        def get_sentence_embedding_dimension(self) -> int:
            return self._dim

        def encode(self, text: str, *args: object, **kwargs: object) -> list[float]:
            t = (text or "").strip()
            if t == "question text":
                return [1.0, 0.0, 0.0, 0.0]
            if "first" in t:
                return [0.0, 1.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0, 0.0]

    fake_mod.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_mod


def test_sentence_transformer_navigator_merge_two_steps() -> None:
    _install_fake_sentence_transformers()

    from src.navigator.sentence_transformer_navigator import (
        SentenceTransformerNavigator,
        SentenceTransformerNavigatorConfig,
    )
    from src.tree_builder import TreeNode

    nav = SentenceTransformerNavigator(
        SentenceTransformerNavigatorConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            dtype="float32",
            max_chars_per_node=512,
            query_cache_max_size=8,
        )
    )
    s0 = nav.init_state()
    n1 = TreeNode(node_id="a", text="first chunk", metadata={})
    s1 = nav.step("question text", n1, s0)
    n2 = TreeNode(node_id="b", text="second chunk", metadata={})
    s2 = nav.step("question text", n2, s1)
    assert len(s2.hidden_summary or []) == 4
    assert s2.relevance_score >= 0.0
