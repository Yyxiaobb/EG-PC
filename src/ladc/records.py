"""Serializable records shared by compression and inference experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LatencyBreakdown:
    """Latency components for one request, measured in milliseconds."""

    compression_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0

    @property
    def end_to_end_ms(self) -> float:
        return self.compression_ms + self.prefill_ms + self.decode_ms

    def to_dict(self) -> dict[str, float]:
        return {
            "compression_ms": self.compression_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "end_to_end_ms": self.end_to_end_ms,
        }


@dataclass(frozen=True)
class CompressionRecord:
    """Output of a compressor before the generation backend is called."""

    method: str
    original_prompt_tokens: int
    compressed_prompt_tokens: int
    compressed_prompt: str
    latency_ms: float
    target_budget_ratio: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_reduction_rate(self) -> float:
        if self.original_prompt_tokens <= 0:
            return 0.0
        return 1.0 - self.compressed_prompt_tokens / self.original_prompt_tokens


@dataclass(frozen=True)
class InferenceRecord:
    """Output and timing information returned by a generation backend."""

    output_text: str
    input_tokens: int
    output_tokens: int
    prefill_ms: float
    decode_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestRecord:
    """One comparable end-to-end experiment row."""

    request_id: str
    method: str
    original_prompt_tokens: int
    compressed_prompt_tokens: int
    output_tokens: int
    quality_score: float | None
    baseline_quality_score: float | None
    latency: LatencyBreakdown
    target_budget_ratio: float | None = None
    quality_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def quality_drop(self) -> float | None:
        if self.quality_score is None or self.baseline_quality_score is None:
            return None
        return self.baseline_quality_score - self.quality_score

    @property
    def token_reduction_rate(self) -> float:
        if self.original_prompt_tokens <= 0:
            return 0.0
        return 1.0 - self.compressed_prompt_tokens / self.original_prompt_tokens

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["latency"] = self.latency.to_dict()
        result["quality_drop"] = self.quality_drop
        result["token_reduction_rate"] = self.token_reduction_rate
        return result
