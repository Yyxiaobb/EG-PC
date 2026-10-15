from ladc.compressors import SessionEvidenceGraphCompressor
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import WorkloadRequest


def _request(request_id: str, session_id: str) -> WorkloadRequest:
    context = "[Film] Blue Horizon was directed by Bob Stone."
    return WorkloadRequest(
        request_id=request_id,
        prompt=f"Context:\n{context}\nQuestion: Who directed Blue Horizon?\nAnswer:",
        question="Who directed Blue Horizon?",
        context=context,
        references=("Bob Stone",),
        metadata={"dataset": "toy", "session_id": session_id},
    )


def test_session_graphs_are_isolated():
    compressor = SessionEvidenceGraphCompressor()
    tokenizer = WhitespaceTokenizer()
    compressor.compress_request(_request("a", "session-a"), tokenizer=tokenizer)
    compressor.compress_request(_request("b", "session-b"), tokenizer=tokenizer)

    assert set(compressor.session_ids()) == {"session-a", "session-b"}
    assert compressor.graph_for_session("session-a") is not compressor.graph_for_session("session-b")
    assert len(compressor.graph_for_session("session-a").nodes) == len(compressor.graph_for_session("session-b").nodes)
    compressor.close()
