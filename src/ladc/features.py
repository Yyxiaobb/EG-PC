"""Cheap request features available before generation starts."""

from __future__ import annotations

from dataclasses import dataclass

from ladc.interfaces import Tokenizer


@dataclass(frozen=True)
class PromptFeatures:
    prompt_tokens: int
    unique_token_ratio: float
    context_query_overlap: float
    question_tokens: int
    query_alignment_position: float
    tail_query_overlap: float
    context_tokens: int

    @property
    def redundancy_ratio(self) -> float:
        return 1.0 - self.unique_token_ratio


def extract_prompt_features(prompt: str, *, tokenizer: Tokenizer) -> PromptFeatures:
    tokens = prompt.split()
    token_count = tokenizer.count_tokens(prompt)
    if not tokens or token_count <= 0:
        return PromptFeatures(0, 0.0, 0.0, 0, 0.5, 0.0, 0)

    normalized = [token.lower().strip(".,:!?()[]") for token in tokens]
    unique_ratio = len(set(normalized)) / len(normalized)
    question_marker = "question"
    markers = [index for index, token in enumerate(normalized) if token == question_marker]
    question_start = markers[-1] + 1 if markers else len(normalized)
    question_tokens = normalized[question_start:]
    context_tokens = normalized[: question_start - 1] if markers else normalized
    question_set = set(question_tokens)
    context_set = set(context_tokens)
    overlap = len(question_set & context_set) / max(1, len(question_set))
    aligned_positions = [
        index / max(1, len(context_tokens) - 1)
        for index, token in enumerate(context_tokens)
        if token in question_set
    ]
    alignment_position = sum(aligned_positions) / len(aligned_positions) if aligned_positions else 0.5
    tail_start = int(len(context_tokens) * 0.75)
    tail_overlap = sum(token in question_set for token in context_tokens[tail_start:]) / max(1, len(question_set))
    return PromptFeatures(
        prompt_tokens=token_count,
        unique_token_ratio=unique_ratio,
        context_query_overlap=overlap,
        question_tokens=len(question_tokens),
        query_alignment_position=alignment_position,
        tail_query_overlap=min(1.0, tail_overlap),
        context_tokens=len(context_tokens),
    )
