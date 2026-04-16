from __future__ import annotations

import unittest

try:
    import torch
except BaseException:  # DLL load / platform issues should not break collection
    torch = None  # type: ignore[assignment]

from src.navigator.mamba2_adapter import (
    Mamba2Navigator,
    Mamba2RuntimeConfig,
    _extract_last_hidden,
)


class Mamba2AdapterUtilsTest(unittest.TestCase):
    def test_extract_last_hidden_from_tensor(self) -> None:
        if torch is None:
            self.skipTest("torch not importable in this environment")
        hidden = torch.randn(2, 5, 8)
        out = _extract_last_hidden(hidden)
        self.assertEqual(out.shape, (2, 8))
        self.assertTrue(torch.allclose(out, hidden[:, -1, :]))

    def test_extract_last_hidden_from_model_output_like(self) -> None:
        if torch is None:
            self.skipTest("torch not importable in this environment")

        class _Out:
            def __init__(self, t: object) -> None:
                self.last_hidden_state = t

        hidden = torch.randn(1, 3, 16)
        out = _extract_last_hidden(_Out(hidden))
        self.assertEqual(out.shape, (1, 16))

    def test_extract_last_hidden_rejects_bad_rank(self) -> None:
        if torch is None:
            self.skipTest("torch not importable in this environment")
        with self.assertRaises(ValueError):
            _extract_last_hidden(torch.randn(1, 8))

    def test_query_cache_lru_and_clear_cache(self) -> None:
        nav = Mamba2Navigator(Mamba2RuntimeConfig(query_cache_max_size=2))
        calls = {"n": 0}

        def fake_encode(text: str, previous: list[float] | None) -> list[float]:
            calls["n"] += 1
            return [float(ord(text[0]))]

        nav._encode_text = fake_encode  # type: ignore[method-assign]

        nav._encode_query("a")
        nav._encode_query("b")
        nav._encode_query("c")
        nav._encode_query("a")
        self.assertEqual(calls["n"], 4)
        nav._encode_query("c")
        self.assertEqual(calls["n"], 4)

        nav.clear_cache()
        nav._encode_query("a")
        self.assertEqual(calls["n"], 5)


if __name__ == "__main__":
    unittest.main()
