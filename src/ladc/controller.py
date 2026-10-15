"""Runtime budget selection under latency and quality constraints."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class BudgetCandidate:
    """A candidate plan produced by a cheap runtime predictor."""

    method: str
    target_budget_ratio: float
    predicted_quality: float
    predicted_compression_ms: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "method": self.method,
            "target_budget_ratio": self.target_budget_ratio,
            "predicted_quality": self.predicted_quality,
            "predicted_compression_ms": self.predicted_compression_ms,
        }


@dataclass(frozen=True)
class BudgetDecision:
    candidate: BudgetCandidate
    predicted_prefill_ms: float
    predicted_decode_ms: float
    predicted_end_to_end_ms: float
    quality_constraint_satisfied: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "predicted_prefill_ms": self.predicted_prefill_ms,
            "predicted_decode_ms": self.predicted_decode_ms,
            "predicted_end_to_end_ms": self.predicted_end_to_end_ms,
            "quality_constraint_satisfied": self.quality_constraint_satisfied,
            "reason": self.reason,
        }


class DynamicBudgetController:
    """Choose the fastest candidate that satisfies a quality floor."""

    def __init__(
        self,
        *,
        prefill_ms_per_token: float = 0.08,
        decode_ms_per_token: float = 2.0,
        quality_drop_tolerance: float = 0.03,
    ) -> None:
        if prefill_ms_per_token < 0 or decode_ms_per_token < 0:
            raise ValueError("latency coefficients must be non-negative")
        if quality_drop_tolerance < 0:
            raise ValueError("quality_drop_tolerance must be non-negative")
        self.prefill_ms_per_token = prefill_ms_per_token
        self.decode_ms_per_token = decode_ms_per_token
        self.quality_drop_tolerance = quality_drop_tolerance

    def choose(
        self,
        *,
        original_prompt_tokens: int,
        output_tokens: int,
        candidates: list[BudgetCandidate],
        baseline_quality: float | None = None,
    ) -> BudgetDecision:
        if original_prompt_tokens <= 0:
            raise ValueError("original_prompt_tokens must be positive")
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if not candidates:
            raise ValueError("at least one budget candidate is required")

        reference_quality = (
            max(candidate.predicted_quality for candidate in candidates)
            if baseline_quality is None
            else baseline_quality
        )
        quality_floor = reference_quality - self.quality_drop_tolerance

        def estimate(candidate: BudgetCandidate) -> tuple[float, float, float]:
            compressed_tokens = max(1, floor(original_prompt_tokens * candidate.target_budget_ratio))
            prefill_ms = compressed_tokens * self.prefill_ms_per_token
            decode_ms = output_tokens * self.decode_ms_per_token
            total_ms = candidate.predicted_compression_ms + prefill_ms + decode_ms
            return prefill_ms, decode_ms, total_ms

        feasible = [candidate for candidate in candidates if candidate.predicted_quality >= quality_floor]
        if feasible:
            chosen = min(feasible, key=lambda candidate: estimate(candidate)[2])
            satisfied = True
            reason = "minimum predicted end-to-end latency within quality floor"
        else:
            chosen = max(candidates, key=lambda candidate: (candidate.predicted_quality, -estimate(candidate)[2]))
            satisfied = False
            reason = "no candidate met quality floor; selected highest predicted quality"

        prefill_ms, decode_ms, total_ms = estimate(chosen)
        return BudgetDecision(
            candidate=chosen,
            predicted_prefill_ms=prefill_ms,
            predicted_decode_ms=decode_ms,
            predicted_end_to_end_ms=total_ms,
            quality_constraint_satisfied=satisfied,
            reason=reason,
        )
