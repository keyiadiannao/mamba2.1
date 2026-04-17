from __future__ import annotations

import unittest

from src.evaluation import normalize_reference_for_scoring


class NormalizeReferenceAnswerTest(unittest.TestCase):
    def test_list_joins(self) -> None:
        self.assertEqual(normalize_reference_for_scoring([" a ", "b"]), "a b")

    def test_none_empty(self) -> None:
        self.assertIsNone(normalize_reference_for_scoring(None))
        self.assertIsNone(normalize_reference_for_scoring(""))
        self.assertIsNone(normalize_reference_for_scoring([]))

    def test_str_strips(self) -> None:
        self.assertEqual(normalize_reference_for_scoring("  x  "), "x")


if __name__ == "__main__":
    unittest.main()
