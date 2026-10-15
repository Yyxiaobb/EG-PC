"""Aggregate latency, token and quality metrics without external dependencies."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from ladc.records import RequestRecord


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_records(records: Iterable[RequestRecord]) -> list[dict[str, float | int | str]]:
    rows = list(records)
    grouped: dict[str, list[RequestRecord]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)

    baseline_latency = {
        row.request_id: row.latency.end_to_end_ms
        for row in rows
        if row.method == "no_compression"
    }
    baseline_quality = {
        row.request_id: row.quality_score
        for row in rows
        if row.method == "no_compression" and row.quality_score is not None
    }

    summaries: list[dict[str, float | int | str]] = []
    for method, method_rows in grouped.items():
        latencies = [row.latency.end_to_end_ms for row in method_rows]
        compression_latencies = [row.latency.compression_ms for row in method_rows]
        quality_rows = [row.quality_score for row in method_rows if row.quality_score is not None]
        quality_drops = [
            baseline_quality[row.request_id] - row.quality_score
            for row in method_rows
            if row.request_id in baseline_quality and row.quality_score is not None
        ]
        speedups = [
            baseline_latency[row.request_id] / row.latency.end_to_end_ms
            for row in method_rows
            if row.request_id in baseline_latency and row.latency.end_to_end_ms > 0
        ]
        metric_names = sorted({name for row in method_rows for name in row.quality_metrics})
        summary: dict[str, float | int | str] = {
            "method": method,
            "n": len(method_rows),
            "mean_token_reduction_rate": mean(row.token_reduction_rate for row in method_rows),
            "mean_compression_ms": mean(compression_latencies),
            "mean_prefill_ms": mean(row.latency.prefill_ms for row in method_rows),
            "mean_decode_ms": mean(row.latency.decode_ms for row in method_rows),
            "mean_end_to_end_ms": mean(latencies),
            "p50_end_to_end_ms": _percentile(latencies, 0.50),
            "p95_end_to_end_ms": _percentile(latencies, 0.95),
            "mean_quality_score": mean(quality_rows) if quality_rows else 0.0,
            "mean_quality_drop_vs_no_compression": mean(quality_drops) if quality_drops else 0.0,
            "mean_speedup_vs_no_compression": mean(speedups) if speedups else 0.0,
        }
        for name in metric_names:
            values = [row.quality_metrics[name] for row in method_rows if name in row.quality_metrics]
            summary[f"mean_{name}"] = mean(values)
        summaries.append(summary)
    return summaries
