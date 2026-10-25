from dataclasses import replace

from ladc.compressors import (
    QualityConstrainedGraphRateDistortionCompressor,
    RelationAwareQCGRDCCompressor,
    TripleAwareQCGRDCCompressor,
    ChainAwareQCGRDCCompressor,
    SessionEvidenceGraphCompressor,
)
from ladc.compressors.qc_grdc import _relation_labels, _extract_relation_triples
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


def test_relation_aware_qc_grdc_detects_predicates_and_rewards_matching_evidence():
    assert _relation_labels("Who directed Blue Horizon?") == {"directed"}
    assert _relation_labels("Blue Horizon was directed by Bob Stone.") == {"directed"}
    compressor = RelationAwareQCGRDCCompressor()
    try:
        query_features = compressor._query_features("Who directed Blue Horizon?", [])
        hit = type("Hit", (), {
            "title": "Film", "text": "Blue Horizon was directed by Bob Stone.",
            "source_document_id": "doc", "path": (), "hops": 0, "node_id": "sentence:1",
        })()
        candidate_features = compressor._candidate_features(hit, query_features)
        assert "relation:directed" in candidate_features
        assert compressor._feature_weight("relation:directed", query_terms=query_features, current=True) > 1.0
    finally:
        compressor.close()


def test_triple_and_chain_variants_extract_provenance_and_detect_shared_entity():
    assert _extract_relation_triples("Blue Horizon", "Blue Horizon was directed by Bob Stone.") == (
        ("Blue Horizon", "directed", "Bob Stone"),
    )
    triples = TripleAwareQCGRDCCompressor()
    chain = ChainAwareQCGRDCCompressor()
    try:
        first = type("Hit", (), {"title": "Blue Horizon", "text": "Blue Horizon was directed by Bob Stone.", "source_document_id": "d1", "path": (), "hops": 0, "node_id": "s1"})()
        second = type("Hit", (), {"title": "Bob Stone", "text": "Bob Stone directed Moonfall.", "source_document_id": "d2", "path": (), "hops": 0, "node_id": "s2"})()
        qf = triples._query_features("Which other film did the director direct?", [first, second])
        assert any(x.startswith("triple:") for x in triples._candidate_features(first, qf))
        assert chain._relation_chain_bundles([first, second])
    finally:
        triples.close()
        chain.close()
