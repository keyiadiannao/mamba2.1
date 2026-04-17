from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseRouter, CosineProbeRouter, LearnedClassifierRouter, RuleRouter


def build_router(config: dict[str, Any]) -> BaseRouter:
    routing_mode = str(config.get("routing_mode", "rule")).lower()

    if routing_mode == "rule":
        return RuleRouter(
            lexical_weight=float(config.get("router_lexical_weight", 1.0)),
            cosine_weight=float(config.get("router_cosine_weight", 0.0)),
        )

    if routing_mode == "cosine_probe":
        return CosineProbeRouter()

    if routing_mode == "learned_classifier":
        checkpoint_path = config.get("router_checkpoint_path")
        if not checkpoint_path:
            raise ValueError(
                "Routing mode 'learned_classifier' requires 'router_checkpoint_path' in the config."
            )
        return LearnedClassifierRouter(Path(str(checkpoint_path)))

    raise ValueError(f"Unsupported routing_mode: {routing_mode}")
