import time

import pytest

from ladc.records import CompressionRecord, LatencyBreakdown, RequestRecord
from ladc.timing import measure_ms


def test_latency_breakdown_sums_components():
    latency = LatencyBreakdown(compression_ms=2.0, prefill_ms=3.5, decode_ms=4.5)
    assert latency.end_to_end_ms == 10.0
    assert latency.to_dict()["end_to_end_ms"] == 10.0


def test_compression_record_reports_token_reduction():
    record = CompressionRecord(
        method="fixed",
        original_prompt_tokens=100,
        compressed_prompt_tokens=60,
        compressed_prompt="short",
        latency_ms=1.0,
    )
    assert record.token_reduction_rate == 0.4


def test_request_record_exposes_quality_drop_and_serialization():
    record = RequestRecord(
        request_id="r1",
        method="dynamic",
        original_prompt_tokens=100,
        compressed_prompt_tokens=50,
        output_tokens=10,
        quality_score=0.95,
        baseline_quality_score=0.98,
        latency=LatencyBreakdown(compression_ms=1.0, prefill_ms=2.0, decode_ms=3.0),
    )
    assert record.quality_drop == pytest.approx(0.03)
    assert record.to_dict()["latency"]["end_to_end_ms"] == 6.0


def test_measure_ms_records_a_positive_interval():
    with measure_ms() as measured:
        time.sleep(0.001)
    assert measured and measured[0] > 0.0
