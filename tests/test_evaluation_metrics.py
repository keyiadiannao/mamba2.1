from __future__ import annotations

import unittest

from src.evaluation import answer_f1, exact_match, rouge_l_f1


class EvaluationMetricsTest(unittest.TestCase):
    def test_exact_match_normalizes_case_and_spaces(self) -> None:
        self.assertEqual(exact_match("  Einstein Proposed Relativity ", "einstein proposed relativity"), 1)

    def test_answer_f1_returns_partial_overlap_score(self) -> None:
        score = answer_f1("alpha beta", "alpha gamma beta")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_rouge_l_f1_returns_one_for_identical_strings(self) -> None:
        self.assertEqual(rouge_l_f1("alpha beta gamma", "alpha beta gamma"), 1.0)


if __name__ == "__main__":
    unittest.main()
