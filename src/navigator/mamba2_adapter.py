from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from src.tree_builder import TreeNode

from .base import BaseNavigator, NavigatorState


@dataclass
class Mamba2RuntimeConfig:
    backend: str = "mamba_ssm"
    model_name: str = "mamba2"
    device: str = "cuda"
    dtype: str = "float16"
    max_tokens_per_node: int = 512
    dependency_module: str = "mamba_ssm"


class Mamba2Navigator(BaseNavigator):
    """Integration placeholder for real mamba2 backends."""

    def __init__(self, config: Mamba2RuntimeConfig | None = None) -> None:
        self.config = config or Mamba2RuntimeConfig()
        self._dependency = None

    def init_state(self) -> NavigatorState:
        self._ensure_dependency_available()
        return NavigatorState()

    def step(self, question: str, node: TreeNode, state: NavigatorState) -> NavigatorState:
        self._ensure_dependency_available()
        raise NotImplementedError(
            f"Real Mamba2 forward integration is not implemented yet for backend "
            f"'{self.config.backend}'. This adapter marks the formal insertion point "
            "for Phase A late-stage integration."
        )

    def _ensure_dependency_available(self) -> Any:
        if self._dependency is not None:
            return self._dependency

        try:
            self._dependency = importlib.import_module(self.config.dependency_module)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Navigator backend '{self.config.backend}' requires the optional dependency "
                f"'{self.config.dependency_module}'. Install it in the active environment "
                "before running real Mamba2-backed navigation."
            ) from exc

        return self._dependency
