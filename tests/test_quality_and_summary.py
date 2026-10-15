from ladc.analysis import summarize_records
from ladc.compressors import FixedBudgetCompressor, NoCompressionCompressor
from ladc.experiments import run_workload_matrix
from ladc.inference import SimulatedInferenceBackend
from ladc.quality import evaluate_generation
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import build_synthetic_request


def test_generation_quality_metrics():
    metrics = evaluate_generation("the answer", ["answer"])
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0


def test_summary_contains_latency_token_and_quality_metrics():
    tokenizer = WhitespaceTokenizer()
    request = build_synthetic_request("r1", target_tokens=256, tokenizer=tokenizer)
    rows = run_workload_matrix(
        [request],
        compressors=[NoCompressionCompressor(), FixedBudgetCompressor(retain_ratio=0.5)],
        backend=SimulatedInferenceBackend(),
        tokenizer=tokenizer,
    )
    summaries = summarize_records(rows)
    assert {row["method"] for row in summaries} == {"no_compression", "fixed_budget"}
    for summary in summaries:
        assert "p50_end_to_end_ms" in summary
        assert "p95_end_to_end_ms" in summary
        assert "mean_token_reduction_rate" in summary
        assert "mean_quality_drop_vs_no_compression" in summary
