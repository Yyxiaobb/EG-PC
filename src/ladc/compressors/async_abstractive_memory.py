"""Per-session evidence graph with post-answer abstractive memory updates."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from time import perf_counter
from typing import Any

from ladc.compressors.abstractive_fused import LocalAbstractiveSummarizer
from ladc.compressors.baselines import NoCompressionCompressor
from ladc.compressors.fused_graph import FusedEvidenceGraphCompressor
from ladc.compressors.session_graph import SessionEvidenceGraphCompressor, request_session_id
from ladc.evidence_builder import _extract_entities, ingest_request_context
from ladc.evidence_graph import EvidenceSpan
from ladc.interfaces import Tokenizer
from ladc.records import CompressionRecord
from ladc.workloads import WorkloadRequest


class AsyncAbstractiveEvidenceMemoryCompressor:
    """Answer the first turn with raw context, then summarize evidence off-path.

    Each session owns a fused evidence graph. The first request is passed
    through unchanged. After generation, its source passages are ingested and
    summarized on a worker thread. Later requests fuse current source evidence
    with completed source-only summaries from that session. A pending summary
    never blocks a new request. Summary nodes retain the IDs of their original
    source sentences as metadata; generated answers are not stored as evidence.
    """

    name = "evidence_graph_abstractive_async_v3"

    def __init__(
        self,
        summarizer: LocalAbstractiveSummarizer,
        *,
        retain_ratio: float = 0.50,
        session_compressor: SessionEvidenceGraphCompressor | None = None,
        max_workers: int = 1,
    ) -> None:
        self.summarizer = summarizer
        self.retain_ratio = retain_ratio
        self._session = session_compressor or SessionEvidenceGraphCompressor(
            compressor_class=FusedEvidenceGraphCompressor,
            compressor_kwargs={"retain_ratio": retain_ratio},
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ladc-async-summary")
        self._lock = Lock()
        self._seen_sessions: set[str] = set()
        self._last_request_by_session: dict[str, str] = {}
        self._futures: set[Future[Any]] = set()
        self._audits: dict[str, dict[str, Any]] = {}

    def compress_request(self, request: WorkloadRequest, *, tokenizer: Tokenizer,
                         target_budget_ratio: float | None = None) -> CompressionRecord:
        session_id = request_session_id(request)
        with self._lock:
            first_turn = session_id not in self._seen_sessions
            self._seen_sessions.add(session_id)
            previous_request = self._last_request_by_session.get(session_id)
            self._last_request_by_session[session_id] = request.request_id
        if previous_request is not None:
            self.mark_completed_before_next_request(previous_request)
        if first_turn:
            raw = NoCompressionCompressor().compress(request.prompt, tokenizer=tokenizer)
            return CompressionRecord(
                method=self.name,
                original_prompt_tokens=raw.original_prompt_tokens,
                compressed_prompt_tokens=raw.compressed_prompt_tokens,
                compressed_prompt=raw.compressed_prompt,
                latency_ms=raw.latency_ms,
                target_budget_ratio=target_budget_ratio,
                metadata={**raw.metadata, "v3_turn_mode": "raw_first_turn", "selected_sentence_ids": []},
            )
        record = self._session.compress_request(
            request, tokenizer=tokenizer,
            target_budget_ratio=target_budget_ratio if target_budget_ratio is not None else self.retain_ratio,
        )
        return CompressionRecord(
            method=self.name,
            original_prompt_tokens=record.original_prompt_tokens,
            compressed_prompt_tokens=record.compressed_prompt_tokens,
            compressed_prompt=record.compressed_prompt,
            latency_ms=record.latency_ms,
            target_budget_ratio=record.target_budget_ratio,
            metadata={**record.metadata, "v3_turn_mode": "fused_current_and_ready_memory"},
        )

    def compress(self, prompt: str, *, tokenizer: Tokenizer,
                 target_budget_ratio: float | None = None) -> CompressionRecord:
        return NoCompressionCompressor().compress(prompt, tokenizer=tokenizer,
                                                  target_budget_ratio=target_budget_ratio)

    def _persist_summary(self, request: WorkloadRequest, compression: CompressionRecord,
                         tokenizer: Tokenizer, submitted_at: float, source_ids: tuple[str, ...]) -> None:
        request_id = request.request_id
        session_id = request_session_id(request)
        with self._lock:
            audit = self._audits.setdefault(request_id, {})
        started_at = perf_counter()
        try:
            graph = self._session.graph_for_session(session_id)
            selected = set(source_ids)
            with graph._lock:
                nodes = [graph.nodes[node_id] for node_id in sorted(selected)
                         if node_id in graph.nodes and graph.nodes[node_id].node_type == "sentence"
                         and not graph.nodes[node_id].metadata.get("memory_representation")]
            spans = [EvidenceSpan(
                document_id=node.source_document_id or "unknown",
                sentence_index=node.source_sentence_index or 0,
                text=node.text,
                title=str(node.metadata.get("title", "")),
                confidence=node.confidence,
                status=node.status,
            ) for node in nodes]
            evidence_text = "\n".join(
                f"[{span.title}] {span.text}" if span.title else span.text for span in spans)
            summary, model_ms, generated_tokens = self.summarizer.summarize(
                request.question, evidence_text)
            summary = summary.strip()
            # Preserve usable source material when a small summarizer emits no
            # note. This is an explicit fallback, not a successful compression.
            fallback = not summary
            memory_text = summary if summary else evidence_text
            if not memory_text.strip():
                raise ValueError("no source evidence available for asynchronous memory")
            graph.add_compressed_turn(
                f"history:{request_id}", request.question, "", spans,
                tokenizer=tokenizer,
                memory_text=memory_text,
                memory_provenance=[node.node_id for node in nodes],
                memory_source_only=True,
                memory_entities=_extract_entities(memory_text, ""),
            )
            complete_at = perf_counter()
            with graph._lock:
                for node in nodes:
                    current = graph.nodes.get(node.node_id)
                    if current is not None:
                        graph.nodes[node.node_id] = type(current)(
                            node_id=current.node_id, node_type=current.node_type, text=current.text,
                            source_document_id=current.source_document_id,
                            source_sentence_index=current.source_sentence_index, turn_id=current.turn_id,
                            confidence=current.confidence, status=current.status,
                            metadata={**current.metadata, "abstractive_summary_pending": False})
            with self._lock:
                audit.update({
                    "history_compression_pending": False,
                    "history_compression_error": None,
                    "history_compression_ms": model_ms,
                    "history_queue_ms": (started_at - submitted_at) * 1000,
                    "history_total_ms": (complete_at - submitted_at) * 1000,
                    "history_original_evidence_tokens": tokenizer.count_tokens(evidence_text),
                    "history_summary_tokens": tokenizer.count_tokens(memory_text),
                    "history_generated_tokens": generated_tokens,
                    "history_reduction_rate": 1.0 - tokenizer.count_tokens(memory_text) / max(1, tokenizer.count_tokens(evidence_text)),
                    "history_source_sentence_count": len(nodes),
                    "history_summary_fallback_to_raw": fallback,
                    "history_provenance_sentence_ids": [node.node_id for node in nodes],
                    "history_completed_at": complete_at,
                    "history_persist_ms": max(0.0, (complete_at - started_at) * 1000 - model_ms),
                })
        except Exception as exc:
            graph = self._session.graph_for_session(session_id)
            with graph._lock:
                for node_id in source_ids:
                    current = graph.nodes.get(node_id)
                    if current is not None:
                        graph.nodes[node_id] = type(current)(
                            node_id=current.node_id, node_type=current.node_type, text=current.text,
                            source_document_id=current.source_document_id,
                            source_sentence_index=current.source_sentence_index, turn_id=current.turn_id,
                            confidence=current.confidence, status=current.status,
                            metadata={**current.metadata, "abstractive_summary_pending": False,
                                      "abstractive_summary_error": type(exc).__name__})
            with self._lock:
                audit.update({"history_compression_pending": False,
                              "history_compression_error": f"{type(exc).__name__}: {exc}"})

    def _forget(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.discard(future)

    def record_generation(self, request: WorkloadRequest, output_text: str, *,
                          compression: CompressionRecord, tokenizer: Tokenizer) -> None:
        del output_text  # Generated answers are intentionally excluded from evidence memory.
        submitted_at = perf_counter()
        session_id = request_session_id(request)
        graph = self._session.graph_for_session(session_id)
        source_ids = tuple(compression.metadata.get("current_sentence_ids", ()))
        if not source_ids:
            # Turn one is intentionally not compressed before generation;
            # ingest its raw evidence only after the answer has returned.
            _, ids = ingest_request_context(request, graph=graph, turn_id=request.request_id,
                                            status="uncertain")
            source_ids = tuple(ids)
        with graph._lock:
            for node_id in source_ids:
                current = graph.nodes.get(node_id)
                if current is not None and current.node_type == "sentence":
                    graph.nodes[node_id] = type(current)(
                        node_id=current.node_id, node_type=current.node_type, text=current.text,
                        source_document_id=current.source_document_id,
                        source_sentence_index=current.source_sentence_index, turn_id=current.turn_id,
                        confidence=current.confidence, status=current.status,
                        metadata={**current.metadata, "abstractive_summary_pending": True})
        with self._lock:
            self._audits[request.request_id] = {
                "history_compression_pending": True,
                "history_compression_completed_before_next_request": False,
                "history_submitted_at": submitted_at,
                "has_next_request": False,
            }
        future = self._executor.submit(self._persist_summary, request, compression, tokenizer,
                                       submitted_at, source_ids)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._forget)

    def mark_completed_before_next_request(self, request_id: str) -> None:
        with self._lock:
            audit = self._audits.get(request_id)
            if audit is not None:
                audit["has_next_request"] = True
                audit["history_compression_completed_before_next_request"] = (
                    not audit.get("history_compression_pending", True)
                    and not audit.get("history_compression_error"))

    def history_audit(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._audits.get(request_id, {}))

    def wait_for_history(self, timeout: float | None = None) -> None:
        with self._lock:
            futures = tuple(self._futures)
        for future in futures:
            future.result(timeout=timeout)
        self._session.wait_for_history(timeout=timeout)

    def close(self) -> None:
        self.wait_for_history()
        self._executor.shutdown(wait=True)
        self._session.close()

