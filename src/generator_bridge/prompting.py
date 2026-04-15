from __future__ import annotations


def build_generator_prompt(question: str, evidence_texts: list[str]) -> str:
    evidence_blocks = []
    for index, text in enumerate(evidence_texts, start=1):
        evidence_blocks.append(f"[{index}] {text}")

    joined_evidence = "\n".join(evidence_blocks) if evidence_blocks else "[no evidence found]"
    return (
        "Answer the question using only the provided evidence.\n"
        "Return only the shortest final answer phrase.\n"
        "Do not explain your reasoning.\n"
        "Do not restate the question or quote the evidence.\n"
        "For yes/no questions, answer with only yes or no.\n"
        f"Question: {question}\n"
        f"Evidence:\n{joined_evidence}\n"
        "Answer:"
    )
