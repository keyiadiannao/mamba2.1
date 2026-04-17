from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _feat(lexical: float, cosine: float) -> dict[str, float]:
    return {
        "child_is_leaf": 0.0,
        "cosine_probe": cosine,
        "lexical_overlap": lexical,
        "parent_relevance": 0.0,
        "text_length_tokens": 1.0,
    }


class TrainLearnedRouterListwiseTest(unittest.TestCase):
    def test_listwise_prefers_high_lexical_child(self) -> None:
        rows = [
            {
                "sample_id": "toy1",
                "parent_node_id": "root",
                "child_node_id": "good",
                "label": 1,
                "features": _feat(5.0, 0.0),
            },
            {
                "sample_id": "toy1",
                "parent_node_id": "root",
                "child_node_id": "bad",
                "label": 0,
                "features": _feat(0.0, 5.0),
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "train.jsonl"
            out = Path(tmp) / "ckpt.json"
            with inp.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            cmd = [
                sys.executable,
                str(ROOT / "scripts/run_nav/train_learned_router.py"),
                "--input",
                str(inp),
                "--output",
                str(out),
                "--epochs",
                "400",
                "--lr",
                "0.2",
                "--loss",
                "listwise_softmax",
            ]
            subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data.get("loss"), "listwise_softmax")
            w_lex = data["feature_names"].index("lexical_overlap")
            w_cos = data["feature_names"].index("cosine_probe")
            self.assertGreater(data["weights"][w_lex], data["weights"][w_cos])


if __name__ == "__main__":
    unittest.main()
