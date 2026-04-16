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
        load_strategy = str(config.get("navigator_load_strategy", "smoke_block"))
        if load_strategy == "hf_pretrained":
            default_dependency_module = "transformers"
        return Mamba2Navigator(
            Mamba2RuntimeConfig(
                backend=navigator_type,
                model_name=str(config.get("navigator_model_name", "mamba2")),
                load_strategy=load_strategy,
                pretrained_checkpoint=(
                    str(config["navigator_pretrained_checkpoint"])
                    if config.get("navigator_pretrained_checkpoint")
                    else None
                ),
                tokenizer_name=(
                    str(config["navigator_tokenizer_name"])
                    if config.get("navigator_tokenizer_name")
                    else None
                ),
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
                use_ssm_continuity=bool(config.get("navigator_use_ssm_continuity", False)),
                query_cache_max_size=int(config.get("navigator_query_cache_max_size", 2048)),
            )
        )

    raise ValueError(f"Unsupported navigator_type: {navigator_type}")
