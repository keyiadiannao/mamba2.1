from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controller import ControllerConfig, SSGSController
from src.generator_bridge import build_generator_prompt
from src.navigator import MockMambaNavigator
from src.router import RuleRouter
from src.tree_builder import DocumentTree, TreeNode


def build_demo_tree() -> DocumentTree:
    root = TreeNode(
        node_id="root",
        text="physics knowledge index",
        children=[
            TreeNode(
                node_id="branch_newton",
                text="classical mechanics and Newton laws",
                children=[
                    TreeNode(
                        node_id="leaf_newton_1",
                        text="Newtonian mechanics explains force, motion, and gravity.",
                    )
                ],
            ),
            TreeNode(
                node_id="branch_relativity",
                text="Einstein relativity branch",
                children=[
                    TreeNode(
                        node_id="leaf_relativity_1",
                        text="Einstein proposed relativity, including special relativity and general relativity.",
                    ),
                    TreeNode(
                        node_id="leaf_relativity_2",
                        text="Relativity changed modern physics and explains spacetime and gravity.",
                    ),
                ],
            ),
        ],
    )
    return DocumentTree(root=root)


def main() -> None:
    question = "What did Einstein propose in relativity?"
    tree = build_demo_tree()
    controller = SSGSController(
        navigator=MockMambaNavigator(),
        router=RuleRouter(),
        config=ControllerConfig(),
    )
    trace = controller.run(question, tree)
    prompt = build_generator_prompt(question, trace.evidence_texts)

    payload = {
        "question": question,
        "trace": trace.to_dict(),
        "generator_prompt": prompt,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
