from dataclasses import replace

from ladc.compressors import (
    QualityConstrainedGraphRateDistortionCompressor,
    SessionEvidenceGraphCompressor,
)
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import WorkloadRequest


def request(rid, question, context, session="qc"):
    prompt = (
        "Answer the question using only the retrieved context.\n"
        f"Context:\n{context}\nQuestion: {question}\nAnswer:"
    )
    return WorkloadRequest(
        rid,
        prompt,
        question,
        context,
        (),
        metadata={"dataset": "toy", "session_id": session},
    )


def test_qc_grdc_keeps_high_distortion_features_and_question_delimiter():
    compressor = QualityConstrainedGraphRateDistortionCompressor(quality_floor=0.90)
    tokenizer = WhitespaceTokenizer()
    sample = request(
        "a",
        "Where did Alice live in 1999 and was it not Paris?",
        "[Alice] Alice lived in Paris in 1999. Alice visited Rome in 2001. "
        "The archive contains unrelated background material.",
    )
    try:
        row = compressor.compress_request(sample, tokenizer=tokenizer, target_budget_ratio=0.55)
        assert row.compressed_prompt.endswith("Question: Where did Alice live in 1999 and was it not Paris?\nAnswer:")
        assert "1999" in row.compressed_prompt
        assert row.metadata["quality_proxy_ratio"] >= 0.90
        assert row.metadata["budget_expanded_for_quality"] is True
        assert row.compressed_prompt_tokens <= row.metadata["budget_tokens"]
    finally:
        compressor.close()


def test_qc_grdc_preserves_a_multi_hop_bridge_from_session_history():
    session = SessionEvidenceGraphCompressor(
        compressor_class=QualityConstrainedGraphRateDistortionCompressor,
        compressor_kwargs={"quality_floor": 0.90, "max_hops": 3},
    )
    tokenizer = WhitespaceTokenizer()
    first = request("a", "Who directed Blue Horizon?", "[Film] Blue Horizon was directed by Bob Stone.")
    second = request("b", "Where was the director of Blue Horizon born?", "[Bob Stone] Bob Stone was born in Bristol.")
    try:
        session.compress_request(first, tokenizer=tokenizer, target_budget_ratio=1.0)
        row = session.compress_request(second, tokenizer=tokenizer, target_budget_ratio=1.0)
        assert "Bristol" in row.compressed_prompt
        assert row.metadata["path_count_selected"] >= 1
        assert row.metadata["historical_candidate_count"] >= 1
    finally:
        session.close()


def test_qc_grdc_is_deterministic_for_same_graph_and_budget():
    tokenizer = WhitespaceTokenizer()
    sample = request(
        "a",
        "Which city did Ada visit?",
        "[Ada] Ada visited Paris. Ada visited Rome. Ada studied mathematics.",
    )
    first = QualityConstrainedGraphRateDistortionCompressor()
    second = QualityConstrainedGraphRateDistortionCompressor()
    try:
        a = first.compress_request(sample, tokenizer=tokenizer, target_budget_ratio=0.6)
        b = second.compress_request(replace(sample, request_id="other"), tokenizer=tokenizer, target_budget_ratio=0.6)
        assert a.compressed_prompt == b.compressed_prompt
        assert a.metadata["selected_sentence_ids"] == b.metadata["selected_sentence_ids"]
    finally:
        first.close()
        second.close()
