"""Protocols for interchangeable compressors and inference backends."""

from __future__ import annotations

from typing import Protocol

from ladc.records import CompressionRecord, InferenceRecord


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int:
        ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        ...


class PromptCompressor(Protocol):
    name: str

    def compress(
        self,
        prompt: str,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        ...


class InferenceBackend(Protocol):
    name: str

    def generate(self, prompt: str, *, tokenizer: Tokenizer) -> InferenceRecord:
        ...
