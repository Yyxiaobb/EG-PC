from dataclasses import replace

from ladc.compressors import EvidenceGraphCompressor, RiskAwareEvidenceGraphCompressor
from ladc.evidence_builder import build_evidence_graph
from ladc.experiments import run_request
from ladc.inference import SimulatedInferenceBackend
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import WorkloadRequest


def _request(request_id: str, question: str) -> WorkloadRequest:
    context = (
        "[Film] Alice starred in Blue Horizon. Blue Horizon was directed by Bob Stone.\n"
        "[City] Blue Horizon premiered in Paris.\n"
        "[Noise] This paragraph contains unrelated filler information."
    )
    prompt = f"Answer the question using only the retrieved context.\nContext:\n{context}\nQuestion: {question}\nAnswer:"
    return WorkloadRequest(
        request_id=request_id,
        prompt=prompt,
        question=question,
        context=context,
        references=("Bob Stone",),
        metadata={"dataset": "toy", "context_titles": ("Film", "City", "Noise")},
    )


def test_graph_compressor_selects_relevant_evidence_and_respects_budget():
    tokenizer = WhitespaceTokenizer()
    request = _request("q1", "Who directed Blue Horizon?")
    compressor = EvidenceGraphCompressor()

    result = compressor.compress_request(
        request,
        tokenizer=tokenizer,
        target_budget_ratio=0.65,
    )

    assert result.method == "ladc_evidence_graph"
    assert result.compressed_prompt_tokens <= result.original_prompt_tokens * 0.65
    assert "Bob Stone" in result.compressed_prompt
    assert result.metadata["selected_evidence_count"] >= 1
    assert result.metadata["candidate_evidence_count"] >= result.metadata["selected_evidence_count"]


def test_second_request_can_reuse_persistent_graph_evidence():
    tokenizer = WhitespaceTokenizer()
    first = _request("q1", "Who directed Blue Horizon?")
    second = _request("q2", "Where did Blue Horizon premiere?")
    # Paris must be historical-only; repeated current sources are not reuse.
    context = second.context.replace("[City] Blue Horizon premiered in Paris.\n", "")
    second = replace(second, context=context, prompt=second.prompt.replace(second.context, context))
    compressor = EvidenceGraphCompressor()
    build_evidence_graph([first], graph=compressor.graph)

    result = compressor.compress_request(second, tokenizer=tokenizer)

    assert result.metadata["historical_candidate_count"] > 0
    assert result.metadata["reused_evidence_count"] > 0
    assert "Paris" in result.compressed_prompt


def test_history_gate_rejects_lexical_only_matches_from_unrelated_context():
    tokenizer = WhitespaceTokenizer()
    first = _request("q1", "Who directed Blue Horizon?")
    second = WorkloadRequest(
        request_id="q2",
        prompt="Answer the question using only the retrieved context.\nContext:\n[Valley] Green Valley premiered in Paris.\nQuestion: Where did Green Valley premiere?\nAnswer:",
        question="Where did Green Valley premiere?",
        context="[Valley] Green Valley premiered in Paris.",
        references=("Paris",),
        metadata={"dataset": "toy", "context_titles": ("Valley",)},
    )
    compressor = EvidenceGraphCompressor()
    compressor.compress_request(first, tokenizer=tokenizer)

    result = compressor.compress_request(second, tokenizer=tokenizer)

    assert result.metadata["historical_candidate_count"] == 0
    assert result.metadata["reused_evidence_count"] == 0


def test_risk_aware_gate_rejects_unrelated_history_with_common_words():
    tokenizer = WhitespaceTokenizer()
    first = _request("q1", "Who directed Blue Horizon?")
    second = WorkloadRequest(
        request_id="q2",
        prompt="Answer the question using only the retrieved context.\nContext:\n[Valley] Green Valley premiered in Paris.\nQuestion: Where did Green Valley premiere?\nAnswer:",
        question="Where did Green Valley premiere?",
        context="[Valley] Green Valley premiered in Paris.",
        references=("Paris",),
        metadata={"dataset": "toy", "context_titles": ("Valley",)},
    )
    compressor = RiskAwareEvidenceGraphCompressor()
    compressor.compress_request(first, tokenizer=tokenizer)

    result = compressor.compress_request(second, tokenizer=tokenizer)

    assert result.method == "risk_aware_evidence_graph"
    assert result.metadata["historical_gate_accepted_count"] == 0
    assert result.metadata["reused_evidence_count"] == 0


def test_risk_aware_gate_reuses_history_through_current_query_entity():
    tokenizer = WhitespaceTokenizer()
    first = _request("q1", "Who directed Blue Horizon?")
    second = WorkloadRequest(
        request_id="q2",
        prompt="Answer the question using only the retrieved context.\nContext:\n[City] Blue Horizon premiered in Paris.\nQuestion: What was directed in Blue Horizon?\nAnswer:",
        question="What was directed in Blue Horizon?",
        context="[City] Blue Horizon premiered in Paris.",
        references=("Blue Horizon",),
        metadata={"dataset": "toy", "context_titles": ("City",)},
    )
    compressor = RiskAwareEvidenceGraphCompressor()
    compressor.compress_request(first, tokenizer=tokenizer)

    result = compressor.compress_request(second, tokenizer=tokenizer)

    assert result.metadata["historical_gate_accepted_count"] > 0
    assert result.metadata["reused_evidence_count"] > 0
    assert "directed by Bob Stone" in result.compressed_prompt


def test_runner_uses_request_aware_compressor_without_changing_latency_accounting():
    tokenizer = WhitespaceTokenizer()
    request = _request("q1", "Who directed Blue Horizon?")
    row = run_request(
        request,
        compressor=EvidenceGraphCompressor(),
        backend=SimulatedInferenceBackend(prefill_ms_per_token=1.0, decode_ms_per_token=2.0),
        tokenizer=tokenizer,
    )

    assert row.method == "ladc_evidence_graph"
    assert row.latency.compression_ms >= 0.0
    assert row.latency.prefill_ms == float(row.compressed_prompt_tokens)
    assert row.metadata["compression_metadata"]["selected_evidence_count"] >= 1
