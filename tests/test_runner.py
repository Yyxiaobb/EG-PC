from ladc.compressors import FixedBudgetCompressor, NoCompressionCompressor
from ladc.experiments import run_request, run_workload_matrix
from ladc.inference import SimulatedInferenceBackend
from ladc.quality import evaluate_context_preservation
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import build_synthetic_request


def test_runner_includes_compression_in_end_to_end_latency():
    tokenizer = WhitespaceTokenizer()
    request = build_synthetic_request("r1", target_tokens=512, tokenizer=tokenizer)
    row = run_request(
        request,
        compressor=FixedBudgetCompressor(retain_ratio=0.5),
        backend=SimulatedInferenceBackend(prefill_ms_per_token=1.0, decode_ms_per_token=2.0),
        tokenizer=tokenizer,
        quality_fn=lambda sample, prompt_text, output: evaluate_context_preservation(
            prompt_text, sample.expected_answer
        ),
    )
    assert row.compressed_prompt_tokens == row.original_prompt_tokens // 2
    assert row.latency.prefill_ms == float(row.compressed_prompt_tokens)
    assert row.latency.decode_ms == 32.0
    assert row.latency.end_to_end_ms >= row.latency.prefill_ms + row.latency.decode_ms
    assert row.quality_metrics["answer_presence"] == 0.0


def test_matrix_runs_all_request_combinations():
    tokenizer = WhitespaceTokenizer()
    requests = [build_synthetic_request("r1", target_tokens=256, tokenizer=tokenizer)]
    rows = run_workload_matrix(
        requests,
        compressors=[NoCompressionCompressor(), FixedBudgetCompressor(retain_ratio=0.5)],
        backend=SimulatedInferenceBackend(),
        tokenizer=tokenizer,
    )
    assert len(rows) == 2
    assert {row.method for row in rows} == {"no_compression", "fixed_budget"}
