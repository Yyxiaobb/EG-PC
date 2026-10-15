"""Paired statistical analysis for the real sequence ablation.

This script treats each sequence as the resampling unit because its turns are
not independent.  It uses only standard-library functionality so the analysis
can be reproduced in the minimal project environment.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = {
    "token_reduction_rate": ("token_reduction_rate", False),
    "end_to_end_ms": ("latency.end_to_end_ms", True),
    "compression_ms": ("latency.compression_ms", True),
    "prefill_ms": ("latency.prefill_ms", True),
    "token_f1": ("quality_metrics.token_f1", False),
    "evidence_f1": ("quality_metrics.evidence_f1", False),
    "supporting_fact_recall": ("quality_metrics.supporting_fact_recall", False),
    "supporting_fact_precision": ("quality_metrics.supporting_fact_precision", False),
    "exact_match": ("quality_metrics.exact_match", False),
}


def _value(row: dict[str, Any], path: str) -> float:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict):
            return 0.0
        current = current.get(part, 0.0)
    return float(current or 0.0)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bootstrap_ci(
    sequence_values: dict[str, list[float]],
    *,
    samples: int = 5000,
    seed: int = 20261013,
) -> dict[str, float]:
    """Cluster bootstrap CI for a mean, resampling complete sequences."""

    if not sequence_values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    sequence_ids = sorted(sequence_values)
    observed = mean(value for values in sequence_values.values() for value in values)
    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(sequence_ids) for _ in sequence_ids]
        values = [value for sequence_id in selected for value in sequence_values[sequence_id]]
        bootstrap_means.append(mean(values))
    return {
        "mean": observed,
        "ci95_low": _percentile(bootstrap_means, 0.025),
        "ci95_high": _percentile(bootstrap_means, 0.975),
    }


def _paired_rows(rows: list[dict[str, Any]], left_method: str, right_method: str, kind: str | None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        request_meta = row["metadata"]["request_metadata"]
        if kind is not None and request_meta["sequence_kind"] != kind:
            continue
        key = (str(request_meta["sequence_id"]), int(request_meta["turn_index"]))
        grouped[key][row["method"]] = row
    paired = []
    for (sequence_id, turn_index), methods in sorted(grouped.items()):
        if left_method not in methods or right_method not in methods:
            continue
        paired.append(
            {
                "sequence_id": sequence_id,
                "turn_index": turn_index,
                "left": methods[left_method],
                "right": methods[right_method],
            }
        )
    return paired


def _paired_summary(paired: list[dict[str, Any]], *, left_method: str, right_method: str) -> dict[str, Any]:
    by_metric: dict[str, Any] = {}
    for name, (path, lower_is_better) in METRICS.items():
        sequence_values: dict[str, list[float]] = defaultdict(list)
        for item in paired:
            difference = _value(item["left"], path) - _value(item["right"], path)
            sequence_values[item["sequence_id"]].append(difference)
        result = _bootstrap_ci(sequence_values)
        result["better_for_left"] = result["mean"] < 0 if lower_is_better else result["mean"] > 0
        by_metric[name] = result

    return {
        "left_method": left_method,
        "right_method": right_method,
        "n_requests": len(paired),
        "n_sequences": len({item["sequence_id"] for item in paired}),
        "difference_definition": "left minus right",
        "metrics": by_metric,
    }


def _guardrails(paired: list[dict[str, Any]]) -> dict[str, Any]:
    def fraction(predicate) -> float:
        return sum(predicate(item) for item in paired) / max(1, len(paired))

    return {
        "history_not_slower_fraction": fraction(
            lambda item: _value(item["left"], "latency.end_to_end_ms") <= _value(item["right"], "latency.end_to_end_ms")
        ),
        "history_token_f1_non_decreasing_fraction": fraction(
            lambda item: _value(item["left"], "quality_metrics.token_f1") >= _value(item["right"], "quality_metrics.token_f1")
        ),
        "history_evidence_f1_non_decreasing_fraction": fraction(
            lambda item: _value(item["left"], "quality_metrics.evidence_f1") >= _value(item["right"], "quality_metrics.evidence_f1")
        ),
        "history_quality_within_0_05_fraction": fraction(
            lambda item: _value(item["left"], "quality_metrics.evidence_f1")
            >= _value(item["right"], "quality_metrics.evidence_f1") - 0.05
        ),
    }


def _generic_guardrails(paired: list[dict[str, Any]]) -> dict[str, float]:
    """Paired safety rates without assuming the left method is History."""

    def fraction(predicate) -> float:
        return sum(predicate(item) for item in paired) / max(1, len(paired))

    return {
        "left_not_slower_fraction": fraction(
            lambda item: _value(item["left"], "latency.end_to_end_ms") <= _value(item["right"], "latency.end_to_end_ms")
        ),
        "left_token_f1_non_decreasing_fraction": fraction(
            lambda item: _value(item["left"], "quality_metrics.token_f1") >= _value(item["right"], "quality_metrics.token_f1")
        ),
        "left_evidence_f1_non_decreasing_fraction": fraction(
            lambda item: _value(item["left"], "quality_metrics.evidence_f1") >= _value(item["right"], "quality_metrics.evidence_f1")
        ),
        "left_quality_within_0_05_fraction": fraction(
            lambda item: _value(item["left"], "quality_metrics.evidence_f1")
            >= _value(item["right"], "quality_metrics.evidence_f1") - 0.05
        ),
    }


def _unrelated_cases(paired: list[dict[str, Any]], limit: int = 20) -> dict[str, Any]:
    reuse_cases = []
    quality_regressions = []
    for item in paired:
        history_meta = item["left"]["metadata"]["compression_metadata"]
        reused = int(history_meta.get("reused_evidence_count", 0))
        evidence_delta = _value(item["left"], "quality_metrics.evidence_f1") - _value(item["right"], "quality_metrics.evidence_f1")
        token_delta = _value(item["left"], "quality_metrics.token_f1") - _value(item["right"], "quality_metrics.token_f1")
        latency_delta = _value(item["left"], "latency.end_to_end_ms") - _value(item["right"], "latency.end_to_end_ms")
        base = {
            "sequence_id": item["sequence_id"],
            "turn_index": item["turn_index"],
            "request_id": item["left"]["request_id"],
            "reused_evidence_count": reused,
            "historical_candidate_count": int(history_meta.get("historical_candidate_count", 0)),
            "evidence_f1_delta_history_minus_current": evidence_delta,
            "token_f1_delta_history_minus_current": token_delta,
            "end_to_end_ms_delta_history_minus_current": latency_delta,
            "history_selected_evidence_count": int(history_meta.get("selected_evidence_count", 0)),
            "current_selected_evidence_count": int(item["right"]["metadata"]["compression_metadata"].get("selected_evidence_count", 0)),
        }
        if reused > 0:
            reuse_cases.append(base)
        if evidence_delta < -0.05 or token_delta < -0.05 or latency_delta > 100.0:
            quality_regressions.append(base)
    reuse_cases.sort(key=lambda case: (-case["reused_evidence_count"], case["evidence_f1_delta_history_minus_current"]))
    quality_regressions.sort(key=lambda case: (case["evidence_f1_delta_history_minus_current"], case["token_f1_delta_history_minus_current"]))
    return {
        "n_requests": len(paired),
        "n_with_historical_reuse": len(reuse_cases),
        "reuse_fraction": len(reuse_cases) / max(1, len(paired)),
        "historical_reuse_cases": reuse_cases[:limit],
        "quality_or_latency_regression_count": len(quality_regressions),
        "quality_or_latency_regressions": quality_regressions[:limit],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/real_7b_sequence_ablation_paper_scale.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/real_7b_sequence_ablation_paired_analysis.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    analyses: dict[str, Any] = {}
    for kind in ("related", "unrelated", None):
        label = kind or "all"
        history_current = _paired_rows(rows, "evidence_graph_history", "evidence_graph_current_only", kind)
        analyses[label] = {
            "history_vs_current_only": _paired_summary(
                history_current,
                left_method="evidence_graph_history",
                right_method="evidence_graph_current_only",
            ),
            "guardrails": _guardrails(history_current),
        }
        if kind == "unrelated":
            analyses[label]["error_analysis"] = _unrelated_cases(history_current)

    comparison_specs = (
        ("risk_vs_current_only", "risk_aware_evidence_graph", "evidence_graph_current_only"),
        ("risk_vs_history", "risk_aware_evidence_graph", "evidence_graph_history"),
    )
    comparisons: dict[str, Any] = {}
    for kind in ("related", "unrelated", None):
        label = kind or "all"
        for comparison_name, left_method, right_method in comparison_specs:
            paired = _paired_rows(rows, left_method, right_method, kind)
            comparisons[f"{label}:{comparison_name}"] = {
                "summary": _paired_summary(paired, left_method=left_method, right_method=right_method),
                "guardrails": _generic_guardrails(paired),
            }
            if kind == "unrelated" and comparison_name == "risk_vs_history":
                comparisons[f"{label}:{comparison_name}"]["error_analysis"] = _unrelated_cases(paired)

    output = {
        "input": str(args.input),
        "rows": len(rows),
        "bootstrap": {"samples": 5000, "seed": 20261013, "unit": "sequence"},
        "analyses": analyses,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
