from __future__ import annotations

from typing import Any

from .base import BaseNavigator, MockMambaNavigator
from .mamba2_adapter import Mamba2Navigator, Mamba2RuntimeConfig


def build_navigator(config: dict[str, Any]) -> BaseNavigator:
    navigator_type = str(config.get("navigator_type", "mock")).lower()

    if navigator_type == "mock":
        return MockMambaNavigator()

    if navigator_type in {"mamba2", "mamba2_native", "mamba_ssm"}:
        default_dependency_module = "mamba_ssm" if navigator_type == "mamba_ssm" else navigator_type
        return Mamba2Navigator(
            Mamba2RuntimeConfig(
                backend=navigator_type,
                model_name=str(config.get("navigator_model_name", "mamba2")),
                device=str(config.get("navigator_device", "cuda")),
                dtype=str(config.get("navigator_dtype", "float16")),
                max_tokens_per_node=int(config.get("navigator_max_tokens_per_node", 512)),
                d_model=int(config.get("navigator_d_model", 64)),
                d_state=int(config.get("navigator_d_state", 64)),
                d_conv=int(config.get("navigator_d_conv", 4)),
                expand=int(config.get("navigator_expand", 2)),
                dependency_module=str(
                    config.get("navigator_dependency_module", default_dependency_module)
                ),
            )
        )

    raise ValueError(f"Unsupported navigator_type: {navigator_type}")
