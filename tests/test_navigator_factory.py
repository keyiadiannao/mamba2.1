from __future__ import annotations

import unittest

from src.navigator import Mamba2Navigator, MockMambaNavigator, build_navigator


class NavigatorFactoryTest(unittest.TestCase):
    def test_build_navigator_returns_mock_by_default(self) -> None:
        navigator = build_navigator({})
        self.assertIsInstance(navigator, MockMambaNavigator)

    def test_build_navigator_returns_mamba2_adapter(self) -> None:
        navigator = build_navigator({"navigator_type": "mamba2"})
        self.assertIsInstance(navigator, Mamba2Navigator)

    def test_build_navigator_returns_native_adapter(self) -> None:
        navigator = build_navigator({"navigator_type": "mamba2_native"})
        self.assertIsInstance(navigator, Mamba2Navigator)

    def test_build_navigator_returns_mamba_ssm_adapter(self) -> None:
        navigator = build_navigator({"navigator_type": "mamba_ssm"})
        self.assertIsInstance(navigator, Mamba2Navigator)
        self.assertEqual(navigator.config.backend, "mamba_ssm")

    def test_build_navigator_rejects_unknown_type(self) -> None:
        with self.assertRaises(ValueError):
            build_navigator({"navigator_type": "unknown"})


if __name__ == "__main__":
    unittest.main()
