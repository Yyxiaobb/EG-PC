"""Evidence-graph-aware sentence compression for request-level experiments."""

from __future__ import annotations

from hashlib import sha1
from math import floor
import re
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any

from ladc.evidence_builder import _extract_entities, ingest_request_context
from ladc.evidence_graph import EvidenceGraph, EvidenceHit, EvidenceSpan, _normalize, _tokens
from ladc.interfaces import Tokenizer
from ladc.records import CompressionRecord
from ladc.timing import measure_ms
from ladc.workloads import WorkloadRequest


_QUESTION_RE = re.compile(r"Question:\s*(.*?)(?:\nAnswer:|$)", re.IGNORECASE | re.DOTALL)
_CONTEXT_RE = re.compile(r"Context:\s*(.*?)(?:\nQuestion:|$)", re.IGNORECASE | re.DOTALL)


def _validate_ratio(value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError("budget ratio must be in the interval (0, 1]")


def _request_from_prompt(prompt: str) -> WorkloadRequest:
    context_match = _CONTEXT_RE.search(prompt)
    question_match = _QUESTION_RE.search(prompt)
    context = context_match.group(1).strip() if context_match else prompt
    question = question_match.group(1).strip() if question_match else ""
    request_id = "prompt:" + sha1(prompt.encode("utf-8")).hexdigest()[:16]
    return WorkloadRequest(
        request_id=request_id,
        prompt=prompt,
        question=question,
        context=context,
        references=(),
        metadata={"dataset": "prompt"},
    )


class EvidenceGraphCompressor:
    """Select source sentences using a persistent evidence graph.

    This first version is intentionally training-free.  Its latency cost is a
    lexical graph lookup plus tokenizer accounting, so it can be measured
    honestly against the prefill saving in the existing runner.
    """

    name = "ladc_evidence_graph"

    def __init__(
        self,
        graph: EvidenceGraph | None = None,
        *,
        min_retain_ratio: float = 0.25,
        max_retain_ratio: float = 0.85,
        top_k_candidates: int = 32,
        max_hops: int = 2,
        name: str | None = None,
    ) -> None:
        _validate_ratio(min_retain_ratio)
        _validate_ratio(max_retain_ratio)
        if min_retain_ratio > max_retain_ratio:
            raise ValueError("min_retain_ratio must not exceed max_retain_ratio")
        if top_k_candidates <= 0:
            raise ValueError("top_k_candidates must be positive")
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        self.graph = graph or EvidenceGraph()
        self.min_retain_ratio = min_retain_ratio
        self.max_retain_ratio = max_retain_ratio
        self.top_k_candidates = top_k_candidates
        self.max_hops = max_hops
        self._history_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ladc-history",
        )
        self._history_futures: set[Future[Any]] = set()
        self._history_futures_lock = Lock()
        if name is not None:
            self.name = name

    def record_generation(
        self,
        request: WorkloadRequest,
        output_text: str,
        *,
        compression: CompressionRecord,
        tokenizer: Tokenizer,
    ) -> None:
        """Queue compact history persistence after generation has completed.

        Submission is deliberately non-blocking.  The selected sentence IDs
        come from the prompt compression record, so the background task never
        needs to recompress the full request or inspect evaluation metadata.
        """

        selected_ids = tuple(compression.metadata.get("selected_sentence_ids", ()))
        question = request.question
        answer = output_text

        def persist() -> None:
            spans: list[Any] = []
            with self.graph._lock:
                selected_nodes = [self.graph.nodes.get(node_id) for node_id in selected_ids]
            for node in selected_nodes:
                if node is None or node.node_type != "sentence":
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
            self.graph.add_compressed_turn(
                f"history:{request.request_id}",
                question,
                answer,
                spans,
                tokenizer=tokenizer,
            )

        future = self._history_executor.submit(persist)
        with self._history_futures_lock:
            self._history_futures.add(future)
        future.add_done_callback(self._forget_history_future)

    def _forget_history_future(self, future: Future[Any]) -> None:
        with self._history_futures_lock:
            self._history_futures.discard(future)

    def wait_for_history(self, timeout: float | None = None) -> None:
        """Wait for queued history writes; useful for shutdown and tests."""

        with self._history_futures_lock:
            futures = tuple(self._history_futures)
        for future in futures:
            future.result(timeout=timeout)

    def close(self) -> None:
        """Flush and stop the background history worker."""

        self.wait_for_history()
        self._history_executor.shutdown(wait=True)

    def compress(
        self,
        prompt: str,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        """Compatibility entry point for the existing PromptCompressor API."""

        return self.compress_request(
            _request_from_prompt(prompt),
            tokenizer=tokenizer,
            target_budget_ratio=target_budget_ratio,
        )

    def compress_request(
        self,
        request: WorkloadRequest,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        if not isinstance(request, WorkloadRequest):
            request = _request_from_prompt(request.prompt)
        original_tokens = tokenizer.count_tokens(request.prompt)
        with measure_ms() as measured:
            before_nodes = len(self.graph.nodes)
            _, current_sentence_ids = ingest_request_context(
                request,
                graph=self.graph,
                turn_id=request.request_id,
                status="uncertain",
            )
            current_ids = set(current_sentence_ids)
            hits = self.graph.search(
                request.question,
                query_entities=_extract_entities(request.question, ""),
                top_k=self.top_k_candidates,
                max_hops=self.max_hops,
            )
            # Historical reuse requires a graph path from a query entity. A
            # lexical hit alone is valid for the current request, but is too
            # weak to import evidence from an unrelated previous turn.
            hits, filter_metadata = self._filter_hits(request, hits, current_ids)
            historical_hits = [hit for hit in hits if hit.node_id not in current_ids]
            ratio = self._choose_ratio(
                target_budget_ratio,
                original_tokens=original_tokens,
                historical_hits=historical_hits,
                query=request.question,
            )
            budget = max(1, floor(original_tokens * ratio))
            compressed_prompt, selected_hits, budget_used = self._pack(
                request,
                hits,
                tokenizer=tokenizer,
                budget=budget,
                priority_ids=self._packing_priority_ids(current_ids),
            )
        compressed_tokens = tokenizer.count_tokens(compressed_prompt)
        selected_ids = {hit.node_id for hit in selected_hits}
        reused_ids = selected_ids - current_ids
        metadata: dict[str, Any] = {
            "graph_nodes_before": before_nodes,
            "graph_nodes_after": len(self.graph.nodes),
            "candidate_evidence_count": len(hits),
            "selected_evidence_count": len(selected_hits),
            "historical_candidate_count": len(historical_hits),
            "reused_evidence_count": len(reused_ids),
            "evidence_reuse_rate": len(reused_ids) / max(1, len(selected_hits)),
            "dynamic_budget_ratio": ratio,
            "budget_tokens": budget,
            "packed_evidence_tokens": budget_used,
            "selected_sentence_ids": [hit.node_id for hit in selected_hits],
        }
        metadata.update(filter_metadata)
        return CompressionRecord(
            method=self.name,
            original_prompt_tokens=original_tokens,
            compressed_prompt_tokens=compressed_tokens,
            compressed_prompt=compressed_prompt,
            latency_ms=measured[0],
            target_budget_ratio=target_budget_ratio,
            metadata=metadata,
        )

    def _filter_hits(
        self,
        request: WorkloadRequest,
        hits: list[EvidenceHit],
        current_ids: set[str],
    ) -> tuple[list[EvidenceHit], dict[str, Any]]:
        """Apply the historical-evidence policy before packing.

        The original compressor keeps the historical search behavior for
        backwards-compatible baselines.  Algorithm variants override this
        hook and return both filtered hits and auditable gate metadata.
        """

        filtered = [
            hit
            for hit in hits
            if hit.node_id in current_ids or hit.hops <= self.max_hops
        ]
        return filtered, {"historical_gate_rejected_count": 0}

    def _packing_priority_ids(self, current_ids: set[str]) -> set[str] | None:
        """Return sentence IDs that should receive packing priority."""

        return None

    def _choose_ratio(
        self,
        requested_ratio: float | None,
        *,
        original_tokens: int,
        historical_hits: list[EvidenceHit],
        query: str,
    ) -> float:
        if requested_ratio is not None:
            _validate_ratio(requested_ratio)
            return requested_ratio
        # Reuse lowers the default budget, while longer queries receive a
        # small safety margin because they tend to require more relations.
        reuse_signal = min(1.0, len(historical_hits) / 4.0)
        query_signal = min(1.0, max(1, len(query.split())) / 20.0)
        ratio = self.max_retain_ratio - 0.35 * reuse_signal + 0.10 * query_signal
        if original_tokens < 128:
            ratio = max(ratio, 0.5)
        return max(self.min_retain_ratio, min(self.max_retain_ratio, ratio))

    def _pack(
        self,
        request: WorkloadRequest,
        hits: list[EvidenceHit],
        *,
        tokenizer: Tokenizer,
        budget: int,
        priority_ids: set[str] | None = None,
    ) -> tuple[str, list[EvidenceHit], int]:
        prefix = "Answer the question using only the retrieved context.\nContext:\n"
        suffix = f"\nQuestion: {request.question}\nAnswer:"
        fixed_tokens = tokenizer.count_tokens(prefix + suffix)
        available = max(1, budget - fixed_tokens)
        selected: list[EvidenceHit] = []
        used = 0
        seen_text: set[str] = set()
        # Relevance is the primary key: a short but merely related sentence
        # must not displace the sentence that contains the answer-bearing
        # relation. Token cost is a tie-breaker within similar scores.
        priority_ids = priority_ids or set()
        ranked = sorted(
            hits,
            key=lambda hit: (
                -(hit.score + (0.12 if hit.node_id in priority_ids else 0.0)),
                tokenizer.count_tokens(hit.text),
                hit.hops,
                hit.source_document_id or "",
                hit.source_sentence_index if hit.source_sentence_index is not None else 0,
            ),
        )
        for hit in ranked:
            normalized = hit.text.casefold().strip()
            if normalized in seen_text:
                continue
            line = f"[{hit.title}] {hit.text}" if hit.title else hit.text
            cost = tokenizer.count_tokens(line + "\n")
            if selected and used + cost > available:
                continue
            if not selected and cost > available:
                line = tokenizer.truncate_to_tokens(line, available)
                cost = tokenizer.count_tokens(line)
            if not line.strip() or cost <= 0:
                continue
            selected.append(hit)
            seen_text.add(normalized)
            used += cost
            if used >= available:
                break

        selected.sort(
            key=lambda hit: (
                hit.source_document_id or "",
                hit.source_sentence_index if hit.source_sentence_index is not None else 0,
            )
        )
        context = "\n".join(
            f"[{hit.title}] {hit.text}" if hit.title else hit.text for hit in selected
        )
        compressed_prompt = prefix + context + suffix
        if tokenizer.count_tokens(compressed_prompt) > budget and selected:
            compressed_prompt = tokenizer.truncate_to_tokens(compressed_prompt, budget)
        return compressed_prompt, selected, used


class RiskAwareEvidenceGraphCompressor(EvidenceGraphCompressor):
    """Evidence-graph compression with query-anchored historical reuse.

    This is an algorithmic variant of :class:`EvidenceGraphCompressor`.  It
    treats cross-turn reuse as a risk-bearing action: a historical sentence is
    admitted only when it has an observable bridge to the current request and
    a minimum query overlap.  Current-turn evidence receives a small packing
    priority so historical memory cannot silently displace fresh evidence.
    """

    name = "risk_aware_evidence_graph"

    def __init__(
        self,
        graph: EvidenceGraph | None = None,
        *,
        min_retain_ratio: float = 0.30,
        max_retain_ratio: float = 0.85,
        top_k_candidates: int = 32,
        max_hops: int = 2,
        min_query_overlap: float = 0.10,
        max_historical_fraction: float = 0.50,
        name: str | None = None,
    ) -> None:
        super().__init__(
            graph=graph,
            min_retain_ratio=min_retain_ratio,
            max_retain_ratio=max_retain_ratio,
            top_k_candidates=top_k_candidates,
            max_hops=max_hops,
            name=name or self.name,
        )
        if not 0.0 <= min_query_overlap <= 1.0:
            raise ValueError("min_query_overlap must be in [0, 1]")
        if not 0.0 < max_historical_fraction <= 1.0:
            raise ValueError("max_historical_fraction must be in (0, 1]")
        self.min_query_overlap = min_query_overlap
        self.max_historical_fraction = max_historical_fraction

    def _sentence_entities(self, sentence_id: str) -> set[str]:
        return {
            _normalize(self.graph.nodes[neighbor].text)
            for neighbor in self.graph._adjacency.get(sentence_id, ())
            if neighbor in self.graph.nodes
            and self.graph.nodes[neighbor].node_type == "entity"
        }

    def _current_titles(self, current_ids: set[str]) -> set[str]:
        return {
            _normalize(str(self.graph.nodes[sentence_id].metadata.get("title", "")))
            for sentence_id in current_ids
            if sentence_id in self.graph.nodes
            and self.graph.nodes[sentence_id].metadata.get("title")
        }

    def _current_entities(self, current_ids: set[str]) -> set[str]:
        entities: set[str] = set()
        for sentence_id in current_ids:
            entities.update(self._sentence_entities(sentence_id))
        return entities

    @staticmethod
    def _entity_query_overlap(entity: str, query_terms: set[str]) -> float:
        entity_terms = _tokens(entity)
        if not entity_terms:
            return 0.0
        return len(entity_terms & query_terms) / len(entity_terms)

    def _filter_hits(
        self,
        request: WorkloadRequest,
        hits: list[EvidenceHit],
        current_ids: set[str],
    ) -> tuple[list[EvidenceHit], dict[str, Any]]:
        query_terms = _tokens(request.question)
        query_entities = {
            _normalize(entity)
            for entity in _extract_entities(request.question, "")
            if _normalize(entity)
        }
        current_entities = self._current_entities(current_ids)
        current_titles = self._current_titles(current_ids)
        accepted: list[EvidenceHit] = []
        rejected = 0
        scored_historical: list[tuple[float, EvidenceHit]] = []

        for hit in hits:
            if hit.node_id in current_ids:
                accepted.append(hit)
                continue

            historical_entities = self._sentence_entities(hit.node_id)
            shared_entities = historical_entities & current_entities
            query_entity_bridge = historical_entities & query_entities
            title_bridge = bool(hit.title and _normalize(hit.title) in current_titles)
            lexical_overlap = len(query_terms & _tokens(hit.text)) / max(1, len(query_terms))

            # Generic words can occur in nearly every document.  A current
            # entity is considered a usable bridge only when it contributes
            # query terms; exact title and explicit query-entity bridges are
            # stronger and do not need this fallback.
            query_linked_entities = {
                entity
                for entity in shared_entities
                if self._entity_query_overlap(entity, query_terms) > 0.0
            }
            bridge = bool(title_bridge or query_entity_bridge or query_linked_entities)
            query_link = bool(query_entity_bridge or lexical_overlap >= self.min_query_overlap)
            if not bridge or not query_link:
                rejected += 1
                continue

            gate_score = (
                0.45 * float(title_bridge)
                + 0.35 * min(1.0, len(query_entity_bridge) / 2.0)
                + 0.20 * min(1.0, len(query_linked_entities) / 2.0)
                + 0.20 * lexical_overlap
                + 0.10 * hit.confidence
            )
            scored_historical.append((gate_score, hit))

        # Keep historical memory bounded by candidate count.  This prevents a
        # broad graph match from consuming the entire prompt budget.
        max_historical = max(1, int(self.top_k_candidates * self.max_historical_fraction))
        scored_historical.sort(key=lambda item: (-item[0], -item[1].score, item[1].hops, item[1].node_id))
        accepted.extend(hit for _, hit in scored_historical[:max_historical])
        rejected += max(0, len(scored_historical) - max_historical)
        return accepted, {
            "historical_gate_rejected_count": rejected,
            "historical_gate_accepted_count": len(scored_historical[:max_historical]),
            "historical_gate_query_entity_count": len(query_entities),
            "historical_gate_current_entity_count": len(current_entities),
        }

    def _choose_ratio(
        self,
        requested_ratio: float | None,
        *,
        original_tokens: int,
        historical_hits: list[EvidenceHit],
        query: str,
    ) -> float:
        if requested_ratio is not None:
            _validate_ratio(requested_ratio)
            return requested_ratio
        if not historical_hits:
            return self.max_retain_ratio
        confidence = sum(min(1.0, max(0.0, hit.score)) for hit in historical_hits) / len(historical_hits)
        reuse_signal = min(1.0, len(historical_hits) / 6.0) * confidence
        query_signal = min(1.0, max(1, len(_tokens(query))) / 20.0)
        ratio = self.max_retain_ratio - 0.22 * reuse_signal + 0.08 * query_signal
        if original_tokens < 128:
            ratio = max(ratio, 0.5)
        return max(self.min_retain_ratio, min(self.max_retain_ratio, ratio))

    def _packing_priority_ids(self, current_ids: set[str]) -> set[str]:
        return current_ids
