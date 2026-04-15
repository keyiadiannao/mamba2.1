from __future__ import annotations

from collections import Counter
import re


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def exact_match(prediction: str, reference: str) -> int:
    return int(normalize_text(prediction) == normalize_text(reference))


def answer_f1(prediction: str, reference: str) -> float:
    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0

    dp = [0] * (len(right) + 1)
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, start=1):
            current = dp[index]
            if left_token == right_token:
                dp[index] = previous + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            previous = current
    return dp[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    if not prediction_tokens or not reference_tokens:
        return 0.0

    lcs = _lcs_length(prediction_tokens, reference_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)
