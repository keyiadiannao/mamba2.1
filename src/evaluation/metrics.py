from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, reference: str) -> int:
    return int(normalize_text(prediction) == normalize_text(reference))
