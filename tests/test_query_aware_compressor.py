from ladc.compressors import QueryAwareMMRCompressor
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import WorkloadRequest


def _request() -> WorkloadRequest:
    context = (
        "[Film] Alice starred in Blue Horizon. Blue Horizon was directed by Bob Stone.\n"
        "[Noise] This sentence contains unrelated filler information.\n"
        "[City] Blue Horizon premiered in Paris."
    )
    return WorkloadRequest(
        request_id="q1",
        prompt=f"Context:\n{context}\nQuestion: Who directed Blue Horizon?\nAnswer:",
        question="Who directed Blue Horizon?",
        context=context,
        references=("Bob Stone",),
        metadata={"context_titles": ("Film", "Noise", "City")},
    )


def test_query_aware_mmr_keeps_relevant_sentence_under_budget():
    result = QueryAwareMMRCompressor(retain_ratio=0.65).compress_request(
        _request(),
        tokenizer=WhitespaceTokenizer(),
    )

    assert result.compressed_prompt_tokens <= result.original_prompt_tokens * 0.65
    assert "Bob Stone" in result.compressed_prompt
    assert result.metadata["historical_candidate_count"] == 0


def test_query_aware_mmr_is_deterministic():
    compressor = QueryAwareMMRCompressor(retain_ratio=0.5)
    tokenizer = WhitespaceTokenizer()

    first = compressor.compress_request(_request(), tokenizer=tokenizer)
    second = compressor.compress_request(_request(), tokenizer=tokenizer)

    assert first.compressed_prompt == second.compressed_prompt
