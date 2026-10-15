"""Empirical calibration for runtime quality prediction."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from ladc.features import PromptFeatures, extract_prompt_features
from ladc.interfaces import PromptCompressor, Tokenizer
from ladc.predictor import HeuristicQualityPredictor
from ladc.quality import evaluate_context_preservation
from ladc.workloads import SyntheticRequest, WorkloadRequest


@dataclass(frozen=True)
class CalibrationExample:
    prompt_tokens: int
    redundancy_ratio: float
    context_query_overlap: float
    query_alignment_position: float
    tail_query_overlap: float
    retain_ratio: float
    observed_quality: float


def collect_calibration_examples(
    requests: list[SyntheticRequest | WorkloadRequest],
    *,
    compressor: PromptCompressor,
    tokenizer: Tokenizer,
    retain_ratios: list[float],
) -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
    for request in requests:
        features = extract_prompt_features(request.prompt, tokenizer=tokenizer)
        for retain_ratio in retain_ratios:
            result = compressor.compress(
                request.prompt,
                tokenizer=tokenizer,
                target_budget_ratio=retain_ratio,
            )
            observed = evaluate_context_preservation(result.compressed_prompt, request.expected_answer)
            examples.append(
                CalibrationExample(
                    prompt_tokens=features.prompt_tokens,
                    redundancy_ratio=features.redundancy_ratio,
                    context_query_overlap=features.context_query_overlap,
                    query_alignment_position=features.query_alignment_position,
                    tail_query_overlap=features.tail_query_overlap,
                    retain_ratio=retain_ratio,
                    observed_quality=observed["primary_quality"],
                )
            )
    return examples


def _bucket(value: float, width: float) -> int:
    return int(value / width)


class EmpiricalQualityPredictor(HeuristicQualityPredictor):
    """Lookup-based predictor calibrated on frozen development examples."""

    def __init__(self, *, safety_margin: float = 0.02) -> None:
        if safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        self.safety_margin = safety_margin
        self._table: dict[tuple[int, int, int, int], float] = {}
        self._global_mean = 1.0

    def fit(self, examples: list[CalibrationExample]) -> "EmpiricalQualityPredictor":
        if not examples:
            raise ValueError("at least one calibration example is required")
        grouped: dict[tuple[int, int, int, int], list[float]] = {}
        for example in examples:
            key = (
                min(example.prompt_tokens // 2048, 8),
                _bucket(example.redundancy_ratio, 0.2),
                _bucket(example.context_query_overlap, 0.2),
                _bucket(example.query_alignment_position, 0.2),
                _bucket(example.tail_query_overlap, 0.2),
                round(example.retain_ratio * 100),
            )
            grouped.setdefault(key, []).append(example.observed_quality)
        self._table = {key: mean(values) for key, values in grouped.items()}
        self._global_mean = mean(example.observed_quality for example in examples)
        return self

    def predict_from_features(self, features: PromptFeatures, retain_ratio: float) -> float:
        key = (
            min(features.prompt_tokens // 2048, 8),
            _bucket(features.redundancy_ratio, 0.2),
            _bucket(features.context_query_overlap, 0.2),
            _bucket(features.query_alignment_position, 0.2),
            _bucket(features.tail_query_overlap, 0.2),
            round(retain_ratio * 100),
        )
        if key in self._table:
            estimate = self._table[key]
        else:
            estimate = super().predict_from_features(features, retain_ratio)
        return max(0.0, min(1.0, estimate - self.safety_margin))
