"""Deterministic inference backend for validating experiment accounting."""

from __future__ import annotations

from ladc.interfaces import Tokenizer
from ladc.records import InferenceRecord


class SimulatedInferenceBackend:
    """Approximate prefill/decode latency from token counts.

    This backend does not claim model performance. It provides a stable test
    oracle before a real runtime is connected.
    """

    name = "simulated"

    def __init__(
        self,
        *,
        prefill_ms_per_token: float = 0.08,
        decode_ms_per_token: float = 2.0,
        output_tokens: int = 16,
    ) -> None:
        if prefill_ms_per_token < 0 or decode_ms_per_token < 0:
            raise ValueError("latency coefficients must be non-negative")
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        self.prefill_ms_per_token = prefill_ms_per_token
        self.decode_ms_per_token = decode_ms_per_token
        self.output_tokens = output_tokens

    def generate(self, prompt: str, *, tokenizer: Tokenizer) -> InferenceRecord:
        input_tokens = tokenizer.count_tokens(prompt)
        prefill_ms = input_tokens * self.prefill_ms_per_token
        decode_ms = self.output_tokens * self.decode_ms_per_token
        return InferenceRecord(
            output_text="simulated_output",
            input_tokens=input_tokens,
            output_tokens=self.output_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            metadata={"backend": self.name},
        )
