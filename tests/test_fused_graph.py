from dataclasses import replace
from threading import Event

from ladc.compressors import (AsyncAbstractiveEvidenceMemoryCompressor, FusedEvidenceGraphCompressor,
                              SessionEvidenceGraphCompressor)
from ladc.compressors.post_generation import PostGenerationLLMLingua2MemoryCompressor
from ladc.compressors.llmlingua2 import LLMLingua2Compressor
from ladc.evidence_builder import ingest_request_context
from ladc.evidence_graph import EvidenceGraph
from ladc.records import CompressionRecord
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import WorkloadRequest


def request(rid, question, context):
    return WorkloadRequest(rid, f"Answer the question using only the retrieved context.\nContext:\n{context}\nQuestion: {question}\nAnswer:",
                           question, context, (), metadata={"dataset": "toy", "session_id": "s"})


def test_new_relation_requires_new_material_even_with_related_history():
    tokenizer = WhitespaceTokenizer()
    compressor = FusedEvidenceGraphCompressor()
    try:
        compressor.compress_request(request("a", "Who directed Blue Horizon?",
            "[Film] Blue Horizon was directed by Bob Stone."), tokenizer=tokenizer, target_budget_ratio=1)
        row = compressor.compress_request(request("b", "Where was Bob Stone born?",
            "[Bob Stone] Bob Stone was born in Bristol."), tokenizer=tokenizer, target_budget_ratio=1)
        assert row.metadata["retrieval_audit"]["winner"] == "fused"
        assert "Bristol" in row.compressed_prompt
        assert row.compressed_prompt.endswith("Question: Where was Bob Stone born?\nAnswer:")
    finally:
        compressor.close()


def test_tiny_budget_never_truncates_question():
    compressor = FusedEvidenceGraphCompressor()
    sample = request("a", "Where was the director of Blue Horizon born?",
                     "[Film] Blue Horizon was directed by Bob Stone, who was born in Bristol.")
    try:
        row = compressor.compress_request(sample, tokenizer=WhitespaceTokenizer(), target_budget_ratio=0.01)
        assert row.compressed_prompt.endswith(f"Question: {sample.question}\nAnswer:")
        assert row.compressed_prompt_tokens <= row.metadata["budget_tokens"]
        assert row.metadata["budget_tokens"] > row.metadata["requested_budget_tokens"]
    finally:
        compressor.close()


def test_repeated_source_keeps_current_ids_after_graph_merge():
    sample = request("a", "Who directed Blue Horizon?", "[Film] Blue Horizon was directed by Bob Stone.")
    graph, first = ingest_request_context(sample, graph=EvidenceGraph())
    _, second = ingest_request_context(replace(sample, request_id="b"), graph=graph)
    assert first and first == second


def test_runtime_policy_ignores_gold_annotations():
    sample = request("a", "Where was Bob Stone born?", "[Bob Stone] Bob Stone was born in Bristol.")
    first, second = FusedEvidenceGraphCompressor(), FusedEvidenceGraphCompressor()
    try:
        a = first.compress_request(sample, tokenizer=WhitespaceTokenizer(), target_budget_ratio=1)
        b = second.compress_request(replace(sample, references=("FORBIDDEN_GOLD",),
            supporting_facts=(("FORBIDDEN_DOCUMENT", 99),)), tokenizer=WhitespaceTokenizer(), target_budget_ratio=1)
        assert a.compressed_prompt == b.compressed_prompt
    finally:
        first.close()
        second.close()


class FakeMemory:
    def compress_request(self, sample, *, tokenizer):
        segments = sample.context.splitlines()
        return CompressionRecord("fake", tokenizer.count_tokens(sample.prompt),
            tokenizer.count_tokens(sample.prompt), sample.prompt, 0,
            metadata={"compressed_context_segments": segments})


def test_source_aligned_llmlingua_disables_context_dropping():
    class Engine:
        def compress_prompt(self, segments, **kwargs):
            assert kwargs["use_context_level_filter"] is False
            return {"compressed_prompt": "\n\n".join(segments),
                    "compressed_prompt_list": segments}

    compressor = LLMLingua2Compressor.__new__(LLMLingua2Compressor)
    compressor.retain_ratio = 0.5
    compressor.return_segments = True
    compressor.enforce_budget = False
    compressor.model_path = "fake"
    compressor._compressor = Engine()
    row = compressor.compress_request(request("a", "Where?", "First source.\nSecond source."),
                                      tokenizer=WhitespaceTokenizer())
    assert row.metadata["compressed_context_segments"] == ["First source.", "Second source."]


def test_background_reconciliation_adds_unselected_sources_without_answer_contamination():
    tokenizer = WhitespaceTokenizer()
    compressor = PostGenerationLLMLingua2MemoryCompressor("unused", reconcile_sources=True,
        memory_compressor=FakeMemory(), session_compressor=SessionEvidenceGraphCompressor(
            compressor_class=FusedEvidenceGraphCompressor))
    sample = request("a", "Where was Bob Stone born?",
        "[Bob Stone] Bob Stone was born in Bristol. Bob Stone later lived in Paris. "
        "Bob Stone retired to London. Bob Stone liked music and painting.")
    try:
        row = compressor.compress_request(sample, tokenizer=tokenizer, target_budget_ratio=0.6)
        compressor.record_generation(sample, "FORGEDANSWER", compression=row, tokenizer=tokenizer)
        compressor.wait_for_history()
        audit = compressor.history_audit("a")
        assert "history_compression_error" not in audit
        assert audit["history_supplemented_source_count"] > 0
        assert audit["history_total_ms"] >= audit["history_compression_ms"]
        graph = compressor._session.graph_for_session("s")
        memories = [n for n in graph.nodes.values() if n.metadata.get("source_only_memory")]
        assert memories and all("FORGEDANSWER" not in n.text for n in memories)
        second = compressor.compress_request(replace(sample, request_id="b"),
            tokenizer=tokenizer, target_budget_ratio=1)
        assert "FORGEDANSWER" not in second.compressed_prompt
        assert compressor.history_audit("a")["history_compression_completed_before_next_request"]
    finally:
        compressor.close()


def test_async_abstractive_v3_uses_raw_first_turn_then_persists_source_memory():
    class FakeSummarizer:
        def summarize(self, question, evidence):
            return "Bob Stone was born in Bristol.", 3.0, 7

    tokenizer = WhitespaceTokenizer()
    compressor = AsyncAbstractiveEvidenceMemoryCompressor(FakeSummarizer(), retain_ratio=1.0)
    first = request("a", "Where was Bob Stone born?",
                    "[Bob Stone] Bob Stone was born in Bristol. Bob Stone later lived in Paris.")
    try:
        initial = compressor.compress_request(first, tokenizer=tokenizer, target_budget_ratio=0.5)
        assert initial.metadata["v3_turn_mode"] == "raw_first_turn"
        assert initial.compressed_prompt == first.prompt
        compressor.record_generation(first, "Bristol", compression=initial, tokenizer=tokenizer)
        compressor.wait_for_history()
        audit = compressor.history_audit("a")
        assert not audit["history_compression_error"]
        assert audit["history_summary_tokens"] < audit["history_original_evidence_tokens"]
        graph = compressor._session.graph_for_session("s")
        memory_nodes = [node for node in graph.nodes.values()
                        if node.metadata.get("memory_representation") and node.metadata.get("source_only_memory")]
        assert len(memory_nodes) == 1
        assert memory_nodes[0].metadata["provenance_sentence_ids"]
        assert "Bristol" in memory_nodes[0].text
        assert any(edge.source_id == memory_nodes[0].node_id and edge.relation == "mentions"
                   for edge in graph.edges.values())
        assert "Bristol" not in [node.text for node in graph.nodes.values() if node.node_type == "answer"]

        second_question = "Where was Bob Stone born and where did Bob Stone later live?"
        second_context = "[Bob Stone] " + "Bob Stone later lived in London. " * 8
        second = replace(first, request_id="b", question=second_question, context=second_context,
                         prompt=f"Context: {second_context}\nQuestion: {second_question}\nAnswer:")
        later = compressor.compress_request(second, tokenizer=tokenizer, target_budget_ratio=1.0)
        assert later.metadata["v3_turn_mode"] == "fused_current_and_ready_memory"
        assert later.metadata["reused_evidence_count"] >= 1
        assert "Bristol" in later.compressed_prompt
        assert "London" in later.compressed_prompt
        assert compressor.history_audit("a")["history_compression_completed_before_next_request"]
    finally:
        compressor.close()


def test_async_abstractive_v3_does_not_expose_raw_history_while_summary_is_pending():
    started, release = Event(), Event()

    class SlowSummarizer:
        def summarize(self, question, evidence):
            started.set()
            assert release.wait(timeout=3)
            return "Bob Stone was born in Bristol.", 3.0, 7

    tokenizer = WhitespaceTokenizer()
    compressor = AsyncAbstractiveEvidenceMemoryCompressor(SlowSummarizer(), retain_ratio=1.0)
    first = request("a", "Where was Bob Stone born?", "Bob Stone was born in Bristol.")
    try:
        record = compressor.compress_request(first, tokenizer=tokenizer)
        compressor.record_generation(first, "Bristol", compression=record, tokenizer=tokenizer)
        assert started.wait(timeout=2)
        second_question = "Where was Bob Stone born?"
        second_context = "Bob Stone later lived in London. " * 8
        second = replace(first, request_id="b", question=second_question, context=second_context,
                         prompt=f"Context: {second_context}\nQuestion: {second_question}\nAnswer:")
        row = compressor.compress_request(second, tokenizer=tokenizer, target_budget_ratio=1.0)
        assert row.metadata["historical_candidate_count"] == 0
        release.set()
        compressor.wait_for_history()
        followup = replace(second, request_id="c", context=second_context + "Bob Stone wrote poetry. " * 8,
                           prompt=f"Context: {second_context}Bob Stone wrote poetry. " * 8
                           + f"\nQuestion: {second_question}\nAnswer:")
        next_row = compressor.compress_request(followup, tokenizer=tokenizer, target_budget_ratio=1.0)
        assert next_row.metadata["reused_evidence_count"] >= 1
    finally:
        release.set()
        compressor.close()
