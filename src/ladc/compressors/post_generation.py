"""Session graph retrieval with LLMLingua-2 memory compression after generation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from threading import Lock
from time import perf_counter
from typing import Any

from ladc.compressors.llmlingua2 import LLMLingua2Compressor
from ladc.compressors.session_graph import SessionEvidenceGraphCompressor, request_session_id
from ladc.interfaces import Tokenizer
from ladc.records import CompressionRecord
from ladc.workloads import WorkloadRequest
from ladc.evidence_graph import EvidenceSpan
from ladc.evidence_builder import _extract_entities


class PostGenerationLLMLingua2MemoryCompressor:
    """Use the dual-path session graph for the request and compress memory later.

    The request path delegates to :class:`SessionEvidenceGraphCompressor`, so
    the user-visible latency contains graph retrieval only.  Once generation
    returns, a worker compresses the question, answer, and selected evidence
    with LLMLingua-2, then stores that representation alongside the original
    source spans and their provenance in the session graph.
    """

    name = "evidence_graph_post_llmlingua2"

    def __init__(
        self,
        model_path: str,
        *,
        retain_ratio: float = 0.50,
        device_map: str = "cuda",
        model_config: dict[str, Any] | None = None,
        llmlingua2_config: dict[str, Any] | None = None,
        compressor_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
        session_compressor: SessionEvidenceGraphCompressor | None = None,
        memory_compressor: LLMLingua2Compressor | None = None,
        reconcile_sources: bool = False,
    ) -> None:
        self.name = name or self.name
        self._session = session_compressor or SessionEvidenceGraphCompressor(compressor_kwargs=compressor_kwargs)
        self.reconcile_sources = reconcile_sources
        self._last_session_request: dict[str, str] = {}
        self._memory = memory_compressor or LLMLingua2Compressor(
            model_path,
            retain_ratio=retain_ratio,
            device_map=device_map,
            model_config=model_config,
            llmlingua2_config=llmlingua2_config,
            return_segments=reconcile_sources,
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladc-post-memory")
        self._futures: set[Future[Any]] = set()
        self._futures_lock = Lock()
        self._audits: dict[str, dict[str, Any]] = {}
        self._audits_lock = Lock()

    def compress_request(
        self,
        request: WorkloadRequest,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        if self.reconcile_sources:
            previous = self._last_session_request.get(request_session_id(request))
            if previous is not None:
                self.mark_completed_before_next_request(previous)
        record = self._session.compress_request(
            request,
            tokenizer=tokenizer,
            target_budget_ratio=target_budget_ratio,
        )
        # Preserve the composition name while retaining the retrieval audit.
        return CompressionRecord(
            method=self.name,
            original_prompt_tokens=record.original_prompt_tokens,
            compressed_prompt_tokens=record.compressed_prompt_tokens,
            compressed_prompt=record.compressed_prompt,
            latency_ms=record.latency_ms,
            target_budget_ratio=record.target_budget_ratio,
            metadata={**record.metadata, "request_compressor": record.method},
        )

    def compress(self, prompt: str, *, tokenizer: Tokenizer, target_budget_ratio: float | None = None) -> CompressionRecord:
        return self._session.compress(prompt, tokenizer=tokenizer, target_budget_ratio=target_budget_ratio)

    def _forget(self, future: Future[Any]) -> None:
        with self._futures_lock:
            self._futures.discard(future)

    def _persist(
        self,
        request: WorkloadRequest,
        output_text: str,
        compression: CompressionRecord,
        tokenizer: Tokenizer,
        started_at: float,
    ) -> None:
        if self.reconcile_sources:
            self._persist_reconciled(request, output_text, compression, tokenizer, started_at)
            return
        request_id = request.request_id
        with self._audits_lock:
            audit = self._audits.setdefault(
                request_id,
                {
                    "history_compression_pending": True,
                    "history_compression_completed_before_next_request": False,
                    "history_compression_ms": 0.0,
                },
            )
        try:
            graph = self._session.graph_for_session(request_session_id(request))
            selected_ids = tuple(compression.metadata.get("selected_sentence_ids", ()))
            spans: list[EvidenceSpan] = []
            with graph._lock:
                selected_nodes = [graph.nodes.get(node_id) for node_id in selected_ids]
            for node in selected_nodes:
                if node is None or node.node_type != "sentence":
                    continue
                # A compressed-memory sentence from an older turn is not
                # re-compressed as source evidence; the original source span
                # remains the provenance anchor.
                if node.metadata.get("memory_representation"):
                    continue
                spans.append(
                    EvidenceSpan(
                        document_id=node.source_document_id or "unknown",
                        sentence_index=node.source_sentence_index or 0,
                        text=node.text,
                        title=str(node.metadata.get("title", "")),
                        confidence=node.confidence,
                        status=node.status,
                    )
                )
            evidence_text = "\n".join(
                f"[{span.title}] {span.text}" if span.title else span.text for span in spans
            )
            memory_context = (
                f"Question: {request.question}\n"
                f"Answer: {output_text.strip()}\n"
                f"Evidence:\n{evidence_text}"
            )
            memory_request = WorkloadRequest(
                request_id=f"memory:{request_id}",
                prompt=memory_context,
                question=request.question,
                context=memory_context,
                references=(),
                supporting_facts=(),
                target_output_tokens=request.target_output_tokens,
                metadata={"dataset": request.metadata.get("dataset", ""), "memory_for": request_id},
            )
            started = perf_counter()
            compressed_memory = self._memory.compress_request(memory_request, tokenizer=tokenizer)
            elapsed_ms = (perf_counter() - started) * 1000.0
            graph.add_compressed_turn(
                f"history:{request_id}",
                request.question,
                output_text,
                spans,
                tokenizer=tokenizer,
                memory_text=compressed_memory.compressed_prompt,
                memory_provenance=selected_ids,
            )
            with self._audits_lock:
                audit.update(
                    {
                        "history_compression_pending": False,
                        "history_compression_ms": elapsed_ms,
                        "history_compression_completed_at": perf_counter(),
                        "history_token_reduction_rate": compressed_memory.token_reduction_rate,
                        "history_original_tokens": compressed_memory.original_prompt_tokens,
                        "history_compressed_tokens": compressed_memory.compressed_prompt_tokens,
                        "history_memory_provenance_count": len(spans),
                        "history_started_at": started_at,
                    }
                )
        except Exception as exc:  # pragma: no cover - defensive background path
            with self._audits_lock:
                audit.update({"history_compression_pending": False, "history_compression_error": type(exc).__name__})

    def record_generation(
        self,
        request: WorkloadRequest,
        output_text: str,
        *,
        compression: CompressionRecord,
        tokenizer: Tokenizer,
    ) -> None:
        submitted = perf_counter()
        self._last_session_request[request_session_id(request)] = request.request_id
        with self._audits_lock:
            self._audits[request.request_id] = {
                "history_compression_pending": True,
                "history_compression_completed_before_next_request": False,
                "history_compression_ms": 0.0,
                "history_submitted_at": submitted,
                "has_next_request": False,
            }
        future = self._executor.submit(
            self._persist, request, output_text, compression, tokenizer, submitted
        )
        with self._futures_lock:
            self._futures.add(future)
        future.add_done_callback(self._forget)

    def history_audit(self, request_id: str) -> dict[str, Any]:
        with self._audits_lock:
            return dict(self._audits.get(request_id, {"history_compression_pending": True}))

    def mark_completed_before_next_request(self, request_id: str) -> None:
        with self._audits_lock:
            audit = self._audits.get(request_id)
            if audit is not None:
                audit["has_next_request"] = True
                audit["next_request_started_at"] = perf_counter()
                audit["history_compression_completed_before_next_request"] = (
                    not audit.get("history_compression_pending", True)
                    and not audit.get("history_compression_error")
                )

    def _persist_reconciled(self, request, output_text, compression, tokenizer, submitted):
        """Compress both paths' source evidence and keep generated text separate."""
        started = perf_counter()
        compute_ms = 0.0
        try:
            graph = self._session.graph_for_session(request_session_id(request))
            ids = tuple(compression.metadata.get("reconciliation_sentence_ids", ()))
            with graph._lock:
                nodes = [graph.nodes[node_id] for node_id in ids if node_id in graph.nodes
                         and graph.nodes[node_id].node_type == "sentence"
                         and not graph.nodes[node_id].metadata.get("memory_representation")]
            segments, owners = [], []
            for node in nodes:
                title = str(node.metadata.get("title", ""))
                # Match LLMLingua2Compressor's word chunk size exactly; preserve
                # segment-to-source mapping even when the classifier drops a title.
                words = ((f"[{title}] " if title else "") + node.text).split()
                for index in range(0, len(words), 120):
                    segments.append(" ".join(words[index:index + 120]))
                    owners.append(node)
            evidence_context = "\n".join(segments)
            evidence_request = WorkloadRequest(
                request_id=f"source-memory:{request.request_id}", prompt=evidence_context,
                context=evidence_context, question=request.question, references=())
            clock = perf_counter()
            evidence = self._memory.compress_request(evidence_request, tokenizer=tokenizer)
            compute_ms += (perf_counter() - clock) * 1000
            compressed_segments = evidence.metadata["compressed_context_segments"]
            if len(compressed_segments) != len(owners):
                raise ValueError("compressed segment provenance mismatch")
            interaction = f"Question: {request.question}\nUnverified generated answer: {output_text.strip()}"
            qa_request = WorkloadRequest(request_id=f"qa-memory:{request.request_id}",
                prompt=interaction, context=interaction, question=request.question, references=())
            clock = perf_counter()
            qa = self._memory.compress_request(qa_request, tokenizer=tokenizer)
            compute_ms += (perf_counter() - clock) * 1000
            # This representation is indexed as an interaction, but the fused
            # retriever explicitly rejects it as source evidence.
            graph.add_compressed_turn(f"history:{request.request_id}", request.question,
                output_text, (), tokenizer=tokenizer,
                memory_text="\n".join(qa.metadata["compressed_context_segments"]),
                memory_provenance=ids)
            added = 0
            with graph._lock:
                for index, (text, owner) in enumerate(zip(compressed_segments, owners)):
                    text = text.strip()
                    if not text:
                        continue
                    title = str(owner.metadata.get("title", ""))
                    node_id = graph.add_span(EvidenceSpan(
                        document_id=f"history:source:{request.request_id}", sentence_index=index,
                        text=text, title=title, entities=_extract_entities(text, title),
                        confidence=0.7, status="uncertain"), turn_id=f"history:{request.request_id}")
                    node = graph.nodes[node_id]
                    graph.nodes[node_id] = replace(node, metadata={**node.metadata,
                        "memory_representation": True, "source_only_memory": True,
                        "provenance_sentence_ids": [owner.node_id],
                        "source_document_id": owner.source_document_id})
                    added += 1
            completed = perf_counter()
            with self._audits_lock:
                self._audits[request.request_id].update({
                    "history_compression_pending": False,
                    "history_compression_ms": compute_ms,
                    "history_queue_ms": (started - submitted) * 1000,
                    "history_persist_ms": (completed - started) * 1000 - compute_ms,
                    "history_total_ms": (completed - submitted) * 1000,
                    "history_compression_completed_at": completed,
                    "history_original_tokens": evidence.original_prompt_tokens + qa.original_prompt_tokens,
                    "history_compressed_tokens": tokenizer.count_tokens("\n".join(compressed_segments))
                        + tokenizer.count_tokens("\n".join(qa.metadata["compressed_context_segments"])),
                    "history_memory_provenance_count": len(nodes),
                    "history_supplemented_source_count": len(set(ids) - set(compression.metadata.get("selected_sentence_ids", ()))),
                    "history_source_memory_count": added,
                })
        except Exception as exc:
            with self._audits_lock:
                self._audits[request.request_id].update({"history_compression_pending": False,
                    "history_compression_error": f"{type(exc).__name__}: {exc}"})

    def wait_for_history(self, timeout: float | None = None) -> None:
        with self._futures_lock:
            futures = tuple(self._futures)
        for future in futures:
            future.result(timeout=timeout)
        self._session.wait_for_history(timeout=timeout)

    def close(self) -> None:
        self.wait_for_history()
        self._executor.shutdown(wait=True)
        self._session.close()
