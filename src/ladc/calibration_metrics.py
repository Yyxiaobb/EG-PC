"""Metrics for evaluating runtime quality prediction calibration."""

from __future__ import annotations

from math import sqrt
from statistics import mean

from ladc.features import extract_prompt_features
from ladc.interfaces import Tokenizer
from ladc.workloads import SyntheticRequest, WorkloadRequest


def evaluate_predictions(
    predictor,
    requests: list[SyntheticRequest | WorkloadRequest],
    *,
    tokenizer: Tokenizer,
    retain_ratios: list[float],
    observed: dict[tuple[str, float], float],
    quality_floor: float = 0.97,
) -> dict[str, float | int]:
    errors: list[float] = []
    predicted_feasible = 0
    unsafe = 0
    for request in requests:
        features = extract_prompt_features(request.prompt, tokenizer=tokenizer)
        for ratio in retain_ratios:
            prediction = predictor.predict_from_features(features, ratio)
            actual = observed[(request.request_id, ratio)]
            errors.append(prediction - actual)
            predicted_ok = prediction >= quality_floor
            actual_ok = actual >= quality_floor
            predicted_feasible += int(predicted_ok)
            unsafe += int(predicted_ok and not actual_ok)
    return {
        "n": len(errors),
        "mae": mean(abs(error) for error in errors) if errors else 0.0,
        "rmse": sqrt(mean(error * error for error in errors)) if errors else 0.0,
        "predicted_feasible_rate": predicted_feasible / max(1, len(errors)),
        "unsafe_prediction_rate": unsafe / max(1, predicted_feasible),
    }
