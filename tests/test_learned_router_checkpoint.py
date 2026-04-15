from __future__ import annotations

import unittest
from pathlib import Path

from src.navigator import NavigatorState
from src.router import LearnedClassifierRouter
from src.tree_builder import TreeNode


class LearnedRouterCheckpointTest(unittest.TestCase):
    def test_demo_checkpoint_loads_and_ranks(self) -> None:
        router = LearnedClassifierRouter(Path("configs/router/learned_router_demo.json"))
        parent = TreeNode(node_id="root", text="physics")
        child_good = TreeNode(node_id="relativity", text="Einstein relativity gravity spacetime")
        child_bad = TreeNode(node_id="fruit", text="apple banana orange")
        decision = router.rank_children(
            "What does relativity explain about gravity?",
            parent,
            [child_bad, child_good],
            NavigatorState(),
        )
        self.assertEqual(decision.ordered_children[0].node_id, "relativity")


if __name__ == "__main__":
    unittest.main()
