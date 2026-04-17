"""Normalize dataset reference answers for EM / F1 scoring."""

from __future__ import annotations


def normalize_reference_for_scoring(value: object) -> str | None:
    """Return a single string reference suitable for ``exact_match`` / ``answer_f1``.

    Handles common corpus shapes:
    - ``str`` (strip; empty -> ``None``)
    - ``list`` of strings (or stringifiable items): non-empty parts joined with a single space
    - ``None`` / unsupported empty -> ``None``

    ``run_end_to_end_batch`` must not use ``str(list)`` on list answers (that breaks scoring).
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunk = item.strip()
            else:
                chunk = str(item).strip() if item is not None else ""
            if chunk:
                parts.append(chunk)
        return " ".join(parts) if parts else None
    if isinstance(value, dict):
        for key in ("answer", "text", "value"):
            inner = value.get(key)
            if inner is not None:
                normalized = normalize_reference_for_scoring(inner)
                if normalized:
                    return normalized
        return None
    text = str(value).strip()
    return text or None
