from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter


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
]


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
        return payload
