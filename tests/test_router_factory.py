from __future__ import annotations

import unittest

from src.navigator import NavigatorState
from src.router import CosineProbeRouter, RuleRouter, build_router
from src.tree_builder import TreeNode


class RouterFactoryTest(unittest.TestCase):
    def test_build_router_rule(self) -> None:
        router = build_router({"routing_mode": "rule"})
        self.assertIsInstance(router, RuleRouter)

    def test_build_router_cosine_probe(self) -> None:
        router = build_router({"routing_mode": "cosine_probe"})
        self.assertIsInstance(router, CosineProbeRouter)

    def test_cosine_probe_prefers_semantically_closer_child(self) -> None:
        router = CosineProbeRouter()
        parent = TreeNode(node_id="root", text="physics")
        child_a = TreeNode(node_id="a", text="Einstein relativity spacetime gravity")
        child_b = TreeNode(node_id="b", text="apple banana orange")
        decision = router.rank_children(
            "What is relativity and gravity?",
            parent,
            [child_b, child_a],
            NavigatorState(),
        )
        self.assertEqual(decision.ordered_children[0].node_id, "a")

    def test_rule_router_default_is_lexical_only(self) -> None:
        router = build_router({"routing_mode": "rule"})
        self.assertIsInstance(router, RuleRouter)
        self.assertEqual(router.lexical_weight, 1.0)
        self.assertEqual(router.cosine_weight, 0.0)

    def test_rule_router_blend_breaks_lexical_ties_with_cosine(self) -> None:
        """Equal lexical_overlap -> tie-break by node_id when cosine_weight=0; cosine shifts order."""
        parent = TreeNode(node_id="root", text="index")
        child_a = TreeNode(node_id="a", text="test word")
        child_b = TreeNode(node_id="b", text="test word " + "x " * 30)
        q = "test word"
        only_lex = RuleRouter(lexical_weight=1.0, cosine_weight=0.0).rank_children(
            q, parent, [child_a, child_b], NavigatorState()
        )
        blended = RuleRouter(lexical_weight=1.0, cosine_weight=8.0).rank_children(
            q, parent, [child_a, child_b], NavigatorState()
        )
        self.assertEqual(only_lex.ordered_children[0].node_id, "b")
        self.assertEqual(blended.ordered_children[0].node_id, "a")

    def test_build_router_passes_router_weights(self) -> None:
        router = build_router(
            {
                "routing_mode": "rule",
                "router_lexical_weight": 0.5,
                "router_cosine_weight": 0.25,
            }
        )
        self.assertIsInstance(router, RuleRouter)
        self.assertEqual(router.lexical_weight, 0.5)
        self.assertEqual(router.cosine_weight, 0.25)


if __name__ == "__main__":
    unittest.main()
