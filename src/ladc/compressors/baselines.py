"""Simple, transparent baselines for the first latency experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import floor

from ladc.interfaces import Tokenizer
from ladc.records import CompressionRecord
from ladc.timing import measure_ms


def _validate_ratio(value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError("budget ratio must be in the interval (0, 1]")


class BaselineCompressor(ABC):
    """Common implementation for compressors that preserve prompt text."""

    name: str

    @abstractmethod
    def _compress_text(self, prompt: str, *, tokenizer: Tokenizer, target_budget_ratio: float | None) -> str:
        raise NotImplementedError

    def compress(
        self,
        prompt: str,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        original_tokens = tokenizer.count_tokens(prompt)
        with measure_ms() as measured:
            compressed = self._compress_text(
                prompt,
                tokenizer=tokenizer,
                target_budget_ratio=target_budget_ratio,
            )
        compressed_tokens = tokenizer.count_tokens(compressed)
        return CompressionRecord(
            method=self.name,
            original_prompt_tokens=original_tokens,
            compressed_prompt_tokens=compressed_tokens,
            compressed_prompt=compressed,
            latency_ms=measured[0],
            target_budget_ratio=target_budget_ratio,
        )


class NoCompressionCompressor(BaselineCompressor):
    name = "no_compression"

    def _compress_text(self, prompt: str, *, tokenizer: Tokenizer, target_budget_ratio: float | None) -> str:
        return prompt


class FixedBudgetCompressor(BaselineCompressor):
    """Keep a fixed fraction of each request's original prompt tokens."""

    name = "fixed_budget"

    def __init__(self, retain_ratio: float = 0.5, *, name: str | None = None) -> None:
        _validate_ratio(retain_ratio)
        self.retain_ratio = retain_ratio
        if name is not None:
            self.name = name

    def _compress_text(self, prompt: str, *, tokenizer: Tokenizer, target_budget_ratio: float | None) -> str:
        ratio = self.retain_ratio if target_budget_ratio is None else target_budget_ratio
        _validate_ratio(ratio)
        original_tokens = tokenizer.count_tokens(prompt)
        budget = max(1, floor(original_tokens * ratio))
        return tokenizer.truncate_to_tokens(prompt, budget)


class TruncationCompressor(BaselineCompressor):
    """Keep a fixed absolute token budget, using a prefix truncation policy."""

    name = "truncation"

    def __init__(self, max_tokens: int = 512) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens

    def _compress_text(self, prompt: str, *, tokenizer: Tokenizer, target_budget_ratio: float | None) -> str:
        budget = self.max_tokens
        if target_budget_ratio is not None:
            _validate_ratio(target_budget_ratio)
            budget = max(1, floor(tokenizer.count_tokens(prompt) * target_budget_ratio))
        return tokenizer.truncate_to_tokens(prompt, budget)
