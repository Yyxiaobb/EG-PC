"""Workload request types and deterministic synthetic smoke-test inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from ladc.interfaces import Tokenizer


@dataclass(frozen=True)
class WorkloadRequest:
    """Dataset-independent request consumed by compression experiments.

    ``prompt`` is the only field a compressor or inference backend needs. The
    references and supporting facts are kept separately for evaluation and
    must never be used by a runtime compression policy.
    """

    request_id: str
    prompt: str
    question: str
    context: str
    references: tuple[str, ...]
    supporting_facts: tuple[tuple[str, int], ...] = ()
    target_output_tokens: int = 32
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expected_answer(self) -> str:
        """Compatibility accessor used by the first calibration utilities."""

        return self.references[0] if self.references else ""


@dataclass(frozen=True)
class SyntheticRequest:
    request_id: str
    prompt: str
    expected_answer: str
    difficulty: str
    redundancy: float
    target_output_tokens: int
    answer_position: str = "late"
    metadata: dict[str, object] = field(default_factory=dict)


def _filler(index: int, width: int) -> str:
    words = [
        "context",
        "document",
        "retrieved",
        "evidence",
        "paragraph",
        "entity",
        "relation",
        "system",
        "inference",
        "record",
    ]
    return " ".join(words[(index + offset) % len(words)] for offset in range(width))


def build_synthetic_request(
    request_id: str,
    *,
    target_tokens: int,
    tokenizer: Tokenizer,
    difficulty: str = "medium",
    redundancy: float = 0.35,
    answer_position: str = "late",
    seed: int = 42,
) -> SyntheticRequest:
    """Build a RAG-like prompt with a known answer-bearing span.

    The answer is placed near the end of the context so truncation baselines
    can be tested. The generated text is only for pipeline and latency tests;
    it is not a substitute for benchmark evaluation.
    """

    if target_tokens < 64:
        raise ValueError("target_tokens must be at least 64")
    if not 0.0 <= redundancy <= 1.0:
        raise ValueError("redundancy must be between 0 and 1")
    if answer_position not in {"early", "middle", "late"}:
        raise ValueError("answer_position must be early, middle or late")

    rng = Random(seed)
    answer = f"answer_{rng.randrange(10_000)}"
    query = (
        "Question: Which answer marker is explicitly supported by the final "
        "retrieved evidence?"
    )
    header = (
        "You are a precise question answering system. Use the retrieved context "
        "and return only the answer marker."
    )
    evidence = f"Final evidence: The supported answer marker is {answer}."

    blocks: list[str] = []
    block_index = 0
    while tokenizer.count_tokens(" ".join([header, *blocks, evidence, query])) < target_tokens:
        width = 12 if redundancy < 0.5 else 18
        if redundancy > 0.0 and block_index % 3 == 1:
            text = _filler(0, width)
        else:
            text = _filler(block_index, width)
        blocks.append(f"Retrieved passage {block_index}: {text}.")
        block_index += 1
        if block_index > target_tokens:
            break

    if answer_position == "early":
        prompt = " ".join([header, evidence, *blocks, query])
    elif answer_position == "middle":
        midpoint = len(blocks) // 2
        prompt = " ".join([header, *blocks[:midpoint], evidence, *blocks[midpoint:], query])
    else:
        prompt = " ".join([header, *blocks, evidence, query])
    prompt_tokens = tokenizer.count_tokens(prompt)
    if prompt_tokens > target_tokens:
        prompt = tokenizer.truncate_to_tokens(prompt, target_tokens)

    return SyntheticRequest(
        request_id=request_id,
        prompt=prompt,
        expected_answer=answer,
        difficulty=difficulty,
        redundancy=redundancy,
        target_output_tokens=8 if difficulty == "easy" else 16,
        answer_position=answer_position,
        metadata={
            "target_tokens": target_tokens,
            "actual_tokens": tokenizer.count_tokens(prompt),
            "answer_is_in_original_prompt": answer in prompt,
            "answer_position": answer_position,
        },
    )


def build_length_matrix(
    lengths: list[int],
    *,
    tokenizer: Tokenizer,
    seed: int = 42,
) -> list[SyntheticRequest]:
    """Create one deterministic request for each target prompt length."""

    return [
        build_synthetic_request(
            f"synthetic_{length}",
            target_tokens=length,
            tokenizer=tokenizer,
            seed=seed + index,
        )
        for index, length in enumerate(lengths)
    ]
