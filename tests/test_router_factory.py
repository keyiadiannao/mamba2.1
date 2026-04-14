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


if __name__ == "__main__":
    unittest.main()
