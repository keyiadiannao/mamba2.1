from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any


FROZEN_TRACE_FIELDS = [
    "routing_mode",
    "context_source",
    "batch_id",
    "leaf_indices_required",
    "nav_target_leaf_index",
    "nav_success",
    "visited_leaf_visits_ordered",
    "visited_leaf_indices_deduped",
    "rollback_count",
    "snapshot_stack_max_depth",
    "snapshot_push_count",
    "snapshot_restore_count",
    "nav_wall_time_ms",
    "context_build_error",
    "exact_match",
    "answer_f1",
    "rouge_l_f1",
    "postprocess_mode",
    "postprocess_rule",
    "entity_boost_alpha",
    "question_entity_count",
    "entity_intersection_size",
    "entity_hit_rate",
]

# Scalar / low-volume fields only — no long visit lists or evidence text bodies.
TRACE_FINGERPRINT_FIELDS = [
    "routing_mode",
    "context_source",
    "batch_id",
    "navigator_type",
    "leaf_indices_required",
    "nav_target_leaf_index",
    "nav_success",
    "rollback_count",
    "snapshot_stack_max_depth",
    "snapshot_push_count",
    "snapshot_restore_count",
    "nav_wall_time_ms",
    "context_build_error",
    "exact_match",
    "answer_f1",
    "rouge_l_f1",
    "postprocess_mode",
    "postprocess_rule",
    "entity_boost_alpha",
    "question_entity_count",
    "entity_intersection_size",
    "entity_hit_rate",
    "failure_attribution",
    "generated_answer",
    "generation_error",
]


def _fingerprint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fp: dict[str, Any] = {k: payload.get(k) for k in TRACE_FINGERPRINT_FIELDS}
    fp["_n_evidence_texts"] = len(payload.get("evidence_texts") or [])
    fp["_n_evidence_node_ids"] = len(payload.get("evidence_node_ids") or [])
    fp["_n_context_texts"] = len(payload.get("context_texts") or [])
    fp["_n_context_node_ids"] = len(payload.get("context_node_ids") or [])
    fp["_n_visited_node_ids"] = len(payload.get("visited_node_ids") or [])
    fp["_n_route_decisions"] = len(payload.get("route_decisions") or [])
    fp["_n_event_log"] = len(payload.get("event_log") or [])
    fp["_n_leaf_visits_ordered"] = len(payload.get("visited_leaf_visits_ordered") or [])
    fp["_n_leaf_indices_deduped"] = len(payload.get("visited_leaf_indices_deduped") or [])
    return fp


@dataclass
class TraceRecord:
    routing_mode: str
    context_source: str
    navigator_type: str | None = None
    batch_id: str | None = None
    leaf_indices_required: list[int] = field(default_factory=list)
    nav_target_leaf_index: int | None = None
    nav_success: bool | None = None
    visited_leaf_visits_ordered: list[int] | None = None
    visited_leaf_indices_deduped: list[int] | None = None
    rollback_count: int = 0
    snapshot_stack_max_depth: int = 0
    snapshot_push_count: int = 0
    snapshot_restore_count: int = 0
    nav_wall_time_ms: float | None = None
    context_build_error: str | None = None
    exact_match: int | None = None
    answer_f1: float | None = None
    rouge_l_f1: float | None = None
    generated_answer: str | None = None
    raw_generated_answer: str | None = None
    postprocessed_answer: str | None = None
    generation_error: str | None = None
    postprocess_mode: str | None = None
    postprocess_rule: str | None = None
    entity_boost_alpha: float | None = None
    question_entity_count: int | None = None
    entity_intersection_size: int | None = None
    entity_hit_rate: float | None = None
    context_texts: list[str] = field(default_factory=list)
    context_node_ids: list[str] = field(default_factory=list)
    evidence_texts: list[str] = field(default_factory=list)
    evidence_node_ids: list[str] = field(default_factory=list)
    visited_node_ids: list[str] = field(default_factory=list)
    node_scores: dict[str, float] = field(default_factory=dict)
    route_decisions: list[dict[str, object]] = field(default_factory=list)
    event_log: list[dict[str, object]] = field(default_factory=list)
    failure_attribution: str | None = None

    _start_time: float = field(default_factory=perf_counter, repr=False)

    def finalize(self) -> None:
        self.nav_wall_time_ms = (perf_counter() - self._start_time) * 1000.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("_start_time", None)
        fp = _fingerprint_payload(payload)
        payload["trace_fingerprint_sha256"] = hashlib.sha256(
            json.dumps(fp, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return payload
