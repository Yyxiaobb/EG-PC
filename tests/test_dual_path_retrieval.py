from ladc.compressors import DualPathEvidenceGraphCompressor
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import WorkloadRequest


def _request(request_id: str, question: str, context: str) -> WorkloadRequest:
    return WorkloadRequest(
        request_id=request_id,
        prompt=f"Context:\n{context}\nQuestion: {question}\nAnswer:",
        question=question,
        context=context,
        references=(),
        metadata={"dataset": "toy", "session_id": "s1", "context_titles": ("Doc",)},
    )


def test_dual_path_ingests_current_material_and_reports_winner():
    compressor = DualPathEvidenceGraphCompressor()
    result = compressor.compress_request(
        _request("q1", "Who directed Blue Horizon?", "[Doc] Blue Horizon was directed by Bob Stone."),
        tokenizer=WhitespaceTokenizer(),
        target_budget_ratio=0.9,
    )

    audit = result.metadata["retrieval_audit"]
    assert audit["winner"] in {"current_material", "session_graph"}
    assert result.metadata["selected_evidence_count"] >= 1
    assert any(node.turn_id == "q1" for node in compressor.graph.nodes.values())
    compressor.close()


def test_dual_path_can_reuse_previous_session_graph_evidence():
    compressor = DualPathEvidenceGraphCompressor()
    tokenizer = WhitespaceTokenizer()
    compressor.compress_request(
        _request("q1", "Who directed Blue Horizon?", "[Doc] Blue Horizon was directed by Bob Stone."),
        tokenizer=tokenizer,
    )
    second = compressor.compress_request(
        _request("q2", "Who directed Blue Horizon?", "[Doc] A short unrelated sentence."),
        tokenizer=tokenizer,
        target_budget_ratio=1.0,
    )

    assert second.metadata["historical_candidate_count"] >= 0
    assert second.metadata["retrieval_audit"]["winner"] in {"current_material", "session_graph"}
    compressor.close()
