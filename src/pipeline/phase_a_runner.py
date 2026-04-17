from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.controller import ControllerConfig, SSGSController
from src.evaluation import answer_f1, exact_match, rouge_l_f1
from src.generator_bridge import build_generator_result
from src.navigator import build_navigator
from src.router import build_router
from src.routing.entity_match import (
    compute_entity_hit_rate,
    compute_entity_match_score,
    extract_question_entities,
)
from src.tracing import (
    append_jsonl,
    build_navigation_summary,
    build_registry_row,
    make_run_id,
    write_json,
    write_run_payload,
)
from src.tree_builder import load_tree_from_payload, load_tree_payload


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_controller(config: dict[str, Any]) -> SSGSController:
    return SSGSController(
        navigator=build_navigator(config),
        router=build_router(config),
        config=ControllerConfig(
            routing_mode=str(config.get("routing_mode", "rule")),
            context_source=str(config.get("context_source", "t1_visited_leaves_ordered")),
            max_evidence=int(config.get("max_evidence", 3)),
            min_relevance_score=float(config.get("min_relevance_score", 1.0)),
            max_depth=int(config.get("max_depth", 8)),
            max_nodes=int(config.get("max_nodes", 64)),
            entity_boost_alpha=float(config.get("entity_boost_alpha", 0.0)),
            evidence_max_per_root_child=int(config.get("evidence_max_per_root_child", 0)),
        ),
    )


def _collect_leaf_nodes(tree) -> list[Any]:
    leaf_nodes = [node for node in tree.walk_depth_first() if node.is_leaf]
    return sorted(
        leaf_nodes,
        key=lambda node: (
            int(node.metadata["leaf_index"]) if isinstance(node.metadata.get("leaf_index"), int) else 10**9,
            node.node_id,
        ),
    )


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _extract_entity_key(node_id: str) -> str:
    if "__sent_" in node_id:
        return node_id.split("__sent_", 1)[0]
    return node_id


def _tokenize_content(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "what",
    }
    tokens = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    return {token for token in tokens if token not in stopwords and len(token) > 1}


def _apply_evidence_controls(
    selected_nodes: list[Any],
    question: str,
    config: dict[str, Any],
) -> list[Any]:
    mode = str(config.get("evidence_control_mode", "baseline")).strip().lower()
    if mode != "anti_collapse":
        return selected_nodes

    per_entity_max = max(1, int(config.get("evidence_control_per_entity_max", 2)))
    require_overlap = bool(config.get("evidence_control_require_question_overlap", True))
    question_tokens = _tokenize_content(question)

    filtered: list[Any] = []
    entity_counts: dict[str, int] = {}
    for node in selected_nodes:
        entity_key = _extract_entity_key(node.node_id)
        current_count = entity_counts.get(entity_key, 0)
        if current_count >= per_entity_max:
            continue
        if require_overlap and question_tokens:
            if not (_tokenize_content(node.text) & question_tokens):
                continue
        entity_counts[entity_key] = current_count + 1
        filtered.append(node)

    if filtered:
        return filtered

    # Fallback: enforce entity cap only, to avoid empty contexts.
    fallback: list[Any] = []
    fallback_counts: dict[str, int] = {}
    for node in selected_nodes:
        entity_key = _extract_entity_key(node.node_id)
        current_count = fallback_counts.get(entity_key, 0)
        if current_count >= per_entity_max:
            continue
        fallback_counts[entity_key] = current_count + 1
        fallback.append(node)

    return fallback if fallback else selected_nodes


def _extract_which_options(question: str) -> tuple[str, str] | None:
    lowered = question.lower()
    if " or " not in lowered:
        return None
    question_body = question.strip().rstrip("?").strip()
    left, right = question_body.rsplit(" or ", 1)
    left_option = left.split(",")[-1].strip()
    right_option = right.strip()
    if not left_option or not right_option:
        return None
    return left_option, right_option


def _normalize_choice(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+", value.lower()))


def _postprocess_generated_answer(
    question: str,
    generated_answer: str | None,
    config: dict[str, Any],
) -> tuple[str | None, str, str | None]:
    mode = str(config.get("postprocess_mode", "off")).strip().lower()
    if mode != "constrained" or not generated_answer:
        return generated_answer, mode, None

    answer = generated_answer.strip()
    normalized_answer = _normalize_choice(answer)
    if not normalized_answer:
        return answer, mode, None

    lowered_question = question.strip().lower()
    if re.match(r"^(do|does|did|is|are|was|were|can|could|has|have|had)\b", lowered_question):
        answer_tokens = set(normalized_answer.split())
        if "yes" in answer_tokens:
            return "yes", mode, "force_yesno"
        if "no" in answer_tokens:
            return "no", mode, "force_yesno"
        return answer, mode, None

    options = _extract_which_options(question)
    if options is None:
        return answer, mode, None

    left_option, right_option = options
    left_norm = _normalize_choice(left_option)
    right_norm = _normalize_choice(right_option)
    if left_norm and left_norm in normalized_answer:
        return left_option, mode, "force_binary_choice"
    if right_norm and right_norm in normalized_answer:
        return right_option, mode, "force_binary_choice"
    return answer, mode, None


def _context_build_max_items(config: dict[str, Any], context_max_items: int) -> int:
    """Cap for how many items enter context *before* ``context_select_*`` ranking.

    When ``context_select_mode`` is active (not ``off``), optional
    ``context_select_pool_max_items`` can be set larger than ``context_max_items``
    so ranking (overlap / entity / dedupe) sees more visited leaves while the
    generator still receives at most ``context_select_k`` after selection.
    """
    mode = str(config.get("context_select_mode", "off")).strip().lower()
    if mode in {"", "off", "none"}:
        return context_max_items
    raw = config.get("context_select_pool_max_items")
    if raw is None:
        return context_max_items
    try:
        pool = int(raw)
    except (TypeError, ValueError):
        return context_max_items
    if pool <= 0:
        return context_max_items
    return max(context_max_items, pool)


def _build_context_from_trace(
    tree,
    trace,
    question: str,
    config: dict[str, Any],
    context_source: str,
    max_items: int,
    *,
    leaf_nodes: list[Any],
    leaf_index_map: dict[int, Any],
) -> tuple[list[str], list[str], str | None]:
    if context_source == "t1_visited_leaves_ordered":
        ordered_indices = _dedupe_preserve_order(list(trace.visited_leaf_visits_ordered or []))
        selected_nodes = [leaf_index_map[index] for index in ordered_indices if index in leaf_index_map]
        selected_nodes = _apply_evidence_controls(selected_nodes, question=question, config=config)
        selected_nodes = selected_nodes[:max_items]
        if selected_nodes:
            return [node.text for node in selected_nodes], [node.node_id for node in selected_nodes], None
        selected_node_ids = list(trace.evidence_node_ids[:max_items])
        selected_texts = list(trace.evidence_texts[:max_items])
        return selected_texts, selected_node_ids, None

    if context_source == "oracle_item_leaves":
        required_indices = [index for index in trace.leaf_indices_required if index in leaf_index_map]
        selected_nodes = [leaf_index_map[index] for index in required_indices[:max_items]]
        if not selected_nodes:
            return [], [], "oracle_item_leaves requires non-empty leaf_indices_required."
        return [node.text for node in selected_nodes], [node.node_id for node in selected_nodes], None

    if context_source == "flat_leaf_concat":
        selected_nodes = leaf_nodes[:max_items]
        return [node.text for node in selected_nodes], [node.node_id for node in selected_nodes], None

    return [], [], f"Unsupported context_source: {context_source}"


def _select_context_items(
    context_texts: list[str],
    context_node_ids: list[str],
    question: str,
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    mode = str(config.get("context_select_mode", "off")).strip().lower()
    if mode in {"", "off", "none"}:
        return context_texts, context_node_ids

    select_k = int(config.get("context_select_k", len(context_texts) or 0))
    if select_k <= 0:
        return [], []

    paired = list(zip(context_texts, context_node_ids))
    if mode == "first_k":
        selected = paired[:select_k]
        return [text for text, _ in selected], [node_id for _, node_id in selected]

    if mode == "dedupe_entity_then_k":
        deduped: list[tuple[str, str]] = []
        seen_entities: set[str] = set()
        for text, node_id in paired:
            entity_key = _extract_entity_key(node_id) if node_id else text
            if entity_key in seen_entities:
                continue
            seen_entities.add(entity_key)
            deduped.append((text, node_id))
            if len(deduped) >= select_k:
                break
        return [text for text, _ in deduped], [node_id for _, node_id in deduped]

    if mode == "question_overlap_topk":
        question_tokens = _tokenize_content(question)
        ranked: list[tuple[float, int, str, str]] = []
        for idx, (text, node_id) in enumerate(paired):
            overlap = 0.0
            if question_tokens:
                overlap = float(len(_tokenize_content(text).intersection(question_tokens)))
            # Keep stable ordering by original index for exact ties.
            ranked.append((overlap, -idx, text, node_id))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = ranked[:select_k]
        return [text for _, _, text, _ in selected], [node_id for _, _, _, node_id in selected]

    if mode == "question_entity_match_topk":
        question_entities = extract_question_entities(question)
        question_tokens = _tokenize_content(question)
        ranked_em: list[tuple[float, float, int, str, str]] = []
        for idx, (text, node_id) in enumerate(paired):
            ent_score = compute_entity_match_score(question_entities, text)
            tok_overlap = (
                float(len(_tokenize_content(text).intersection(question_tokens))) if question_tokens else 0.0
            )
            ranked_em.append((ent_score, tok_overlap, -idx, text, node_id))
        ranked_em.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected_em = ranked_em[:select_k]
        return [text for _, _, _, text, _ in selected_em], [node_id for _, _, _, _, node_id in selected_em]

    return context_texts, context_node_ids


def _collect_visited_leaf_texts(trace: Any, leaf_index_map: dict[int, Any]) -> list[str]:
    visited_leaf_texts: list[str] = []
    for leaf_index in list(trace.visited_leaf_visits_ordered or []):
        node = leaf_index_map.get(leaf_index)
        if node is not None:
            visited_leaf_texts.append(node.text)
    if visited_leaf_texts:
        return visited_leaf_texts
    return list(trace.evidence_texts or [])


def run_navigation_sample(
    root_dir: Path,
    config: dict[str, Any],
    question: str,
    tree_path: str,
    reference_answer: str | None = None,
    run_id_prefix: str | None = None,
    sample_id: str | None = None,
    batch_id: str | None = None,
    leaf_indices_required: list[int] | None = None,
    controller: SSGSController | None = None,
) -> dict[str, Any]:
    """单样本 Phase A：导航 → 构建 context → 可选生成与打分。

    在写入 trace 的 context 之前，可按 ``context_select_mode`` / ``context_select_k`` 对证据列表做后处理
    （``off`` / ``first_k`` / ``dedupe_entity_then_k`` / ``question_overlap_topk`` / ``question_entity_match_topk``）；行为说明见 ``docs/Major_Issues_And_Resolutions_CN.md``（MI-006）。
    若 ``context_select_mode`` 非 ``off``，可选用 ``context_select_pool_max_items``（≥ ``context_max_items``）扩大进入打分的候选条数，再截断到 ``context_select_k``。
    """
    resolved_tree_path = root_dir / tree_path
    tree_payload = load_tree_payload(resolved_tree_path)
    tree = load_tree_from_payload(tree_payload)

    final_question = question or str(tree_payload.get("question") or "")
    if not final_question:
        raise ValueError("A question must be provided in the config, sample, or tree payload.")

    active_controller = controller or build_controller(config)
    trace = active_controller.run(final_question, tree)
    trace.entity_boost_alpha = float(config.get("entity_boost_alpha", trace.entity_boost_alpha or 0.0))
    if leaf_indices_required is not None:
        trace.leaf_indices_required = list(leaf_indices_required)
    trace.batch_id = batch_id

    question_entities = extract_question_entities(final_question)
    leaf_nodes = _collect_leaf_nodes(tree)
    leaf_index_map = {
        int(node.metadata["leaf_index"]): node
        for node in leaf_nodes
        if isinstance(node.metadata.get("leaf_index"), int)
    }
    visited_leaf_texts = _collect_visited_leaf_texts(trace, leaf_index_map)
    entity_hit_rate, entity_intersection_size = compute_entity_hit_rate(question_entities, visited_leaf_texts)
    trace.question_entity_count = len(question_entities)
    trace.entity_intersection_size = entity_intersection_size
    trace.entity_hit_rate = entity_hit_rate

    final_reference = reference_answer
    if final_reference is None:
        tree_reference = tree_payload.get("reference_answer")
        final_reference = tree_reference if isinstance(tree_reference, str) else None

    context_max_items = int(config.get("context_max_items", config.get("max_evidence", 3)))
    build_max_items = _context_build_max_items(config, context_max_items)
    context_texts, context_node_ids, context_error = _build_context_from_trace(
        tree=tree,
        trace=trace,
        question=final_question,
        config=config,
        context_source=str(config.get("context_source", "t1_visited_leaves_ordered")),
        max_items=build_max_items,
        leaf_nodes=leaf_nodes,
        leaf_index_map=leaf_index_map,
    )
    context_texts, context_node_ids = _select_context_items(
        context_texts,
        context_node_ids,
        final_question,
        config,
    )
    trace.context_texts = context_texts
    trace.context_node_ids = context_node_ids
    if context_error:
        trace.context_build_error = context_error
        if not trace.failure_attribution:
            trace.failure_attribution = "context_construction_failure"

    run_generator = bool(config.get("run_generator", False))
    if context_error:
        raw_generated_answer, prompt, generation_error = None, None, "skipped_due_to_context_error"
    else:
        raw_generated_answer, prompt, generation_error = build_generator_result(
            config, final_question, context_texts
        )
    postprocessed_answer, postprocess_mode, postprocess_rule = _postprocess_generated_answer(
        question=final_question,
        generated_answer=raw_generated_answer,
        config=config,
    )
    trace.raw_generated_answer = raw_generated_answer
    trace.postprocessed_answer = postprocessed_answer
    trace.generated_answer = postprocessed_answer
    trace.postprocess_mode = postprocess_mode
    trace.postprocess_rule = postprocess_rule
    trace.generation_error = generation_error
    if generation_error and not context_error and not trace.failure_attribution:
        trace.failure_attribution = "generation_failure"

    explicit_eval = str(config.get("eval_mode", "")).strip().lower()
    if explicit_eval in ("generation", "retrieval"):
        eval_mode = explicit_eval
    else:
        eval_mode = "generation" if run_generator else "retrieval"

    if isinstance(final_reference, str):
        if context_error:
            trace.exact_match = 0
            trace.answer_f1 = 0.0
            trace.rouge_l_f1 = 0.0
        elif eval_mode == "generation":
            if not run_generator or generation_error:
                trace.exact_match = 0
                trace.answer_f1 = 0.0
                trace.rouge_l_f1 = 0.0
            else:
                score_source = postprocessed_answer or ""
                trace.exact_match = exact_match(score_source, final_reference)
                trace.answer_f1 = answer_f1(score_source, final_reference)
                trace.rouge_l_f1 = rouge_l_f1(score_source, final_reference)
        else:
            score_source: str | None = None
            if context_texts:
                score_source = context_texts[0]
            elif trace.evidence_texts:
                score_source = trace.evidence_texts[0]

            if isinstance(score_source, str) and score_source:
                trace.exact_match = exact_match(score_source, final_reference)
                trace.answer_f1 = answer_f1(score_source, final_reference)
                trace.rouge_l_f1 = rouge_l_f1(score_source, final_reference)

    run_id = make_run_id(run_id_prefix or str(config.get("run_id_prefix", "phase_a")))
    payload = {
        "run_id": run_id,
        "sample_id": sample_id,
        "batch_id": batch_id,
        "eval_mode": eval_mode,
        "config": config,
        "question": final_question,
        "tree_path": tree_path,
        "trace": trace.to_dict(),
        "generator_prompt": prompt,
        "generator_evidence_texts": context_texts,
        "generator_evidence_node_ids": context_node_ids,
        "raw_generated_answer": raw_generated_answer,
        "postprocessed_answer": postprocessed_answer,
        "generated_answer": postprocessed_answer,
        "reference_answer": final_reference,
    }

    output_dir = root_dir / str(config.get("output_dir", "outputs/runs"))
    output_path = write_run_payload(output_dir, payload, run_id)
    payload["output_run_dir"] = str(output_path.parent)
    write_json(output_path, payload)

    registry_row = build_registry_row(payload)
    navigation_summary = build_navigation_summary(payload)
    write_json(output_path.parent / "registry_row.json", registry_row)
    write_json(output_path.parent / "navigation_summary.json", navigation_summary)
    report_dir = root_dir / str(config.get("report_dir", "outputs/reports"))
    append_jsonl(report_dir / "run_registry.jsonl", registry_row)
    append_jsonl(report_dir / "navigation_summary.jsonl", navigation_summary)

    return payload


def build_batch_summary(batch_id: str, sample_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(sample_payloads)
    nav_successes = sum(1 for payload in sample_payloads if payload["trace"].get("nav_success"))
    exact_match_hits = sum(
        1
        for payload in sample_payloads
        if payload["trace"].get("exact_match") == 1
    )
    total_nav_time = sum(float(payload["trace"].get("nav_wall_time_ms") or 0.0) for payload in sample_payloads)
    answer_f1_values = [
        float(payload["trace"]["answer_f1"])
        for payload in sample_payloads
        if payload["trace"].get("answer_f1") is not None
    ]
    rouge_l_values = [
        float(payload["trace"]["rouge_l_f1"])
        for payload in sample_payloads
        if payload["trace"].get("rouge_l_f1") is not None
    ]

    return {
        "batch_id": batch_id,
        "sample_count": sample_count,
        "nav_success_count": nav_successes,
        "nav_success_rate": (nav_successes / sample_count) if sample_count else 0.0,
        "exact_match_count": exact_match_hits,
        "exact_match_rate": (exact_match_hits / sample_count) if sample_count else 0.0,
        "avg_answer_f1": (sum(answer_f1_values) / len(answer_f1_values)) if answer_f1_values else 0.0,
        "avg_rouge_l_f1": (sum(rouge_l_values) / len(rouge_l_values)) if rouge_l_values else 0.0,
        "avg_nav_wall_time_ms": (total_nav_time / sample_count) if sample_count else 0.0,
        "sample_run_ids": [payload["run_id"] for payload in sample_payloads],
        "sample_ids": [payload.get("sample_id") for payload in sample_payloads],
    }
