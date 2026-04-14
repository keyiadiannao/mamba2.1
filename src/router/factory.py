from __future__ import annotations

from typing import Any

from .base import BaseRouter, CosineProbeRouter, RuleRouter


def build_router(config: dict[str, Any]) -> BaseRouter:
    routing_mode = str(config.get("routing_mode", "rule")).lower()

    if routing_mode == "rule":
        return RuleRouter()

    if routing_mode == "cosine_probe":
        return CosineProbeRouter()

    if routing_mode == "learned_classifier":
        raise NotImplementedError(
            "Routing mode 'learned_classifier' is reserved for a later stage. "
            "Use 'rule' or 'cosine_probe' in the current Phase A navigation framework."
        )

    raise ValueError(f"Unsupported routing_mode: {routing_mode}")
