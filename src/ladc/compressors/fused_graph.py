"""Current-source coverage, bounded history, and provenance-aware fusion."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from hashlib import sha256
from threading import Lock
from typing import Any

from ladc.compressors.evidence_graph import EvidenceGraphCompressor, _validate_ratio
from ladc.compressors.query_aware import QueryAwareMMRCompressor
from ladc.evidence_builder import _extract_entities, ingest_request_context
from ladc.evidence_graph import EvidenceGraph, EvidenceHit, _tokens
from ladc.interfaces import Tokenizer
from ladc.records import CompressionRecord
from ladc.timing import measure_ms
from ladc.workloads import WorkloadRequest


class FusedEvidenceGraphCompressor(EvidenceGraphCompressor):
    """Fuse both paths unless the exact current sources are already cached.

    Source coverage is a conservative prerequisite for the history shortcut,
    not a semantic proof that an answer is entailed. Raw spans remain available
    as provenance anchors. Generated answers are never factual candidates.
    """

    name = "evidence_graph_fused_v2"

    def __init__(self, *args: Any, retain_ratio: float = 0.50,
                 max_history_fraction: float = 0.20, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _validate_ratio(retain_ratio)
        if not 0 <= max_history_fraction <= 0.5:
            raise ValueError("max_history_fraction must be between 0 and 0.5")
        self.retain_ratio = retain_ratio
        self.max_history_fraction = max_history_fraction
        self._retrieval_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ladc-fused")
        self._retrieval_futures: list = []
        self._cache: dict[str, set[str]] = {}
        self._state_lock = Lock()
        self._ranker = QueryAwareMMRCompressor()

    def _search_current(self, request: WorkloadRequest):
        graph, ids = ingest_request_context(request, graph=EvidenceGraph(), status="uncertain")
        return graph.search(request.question, query_entities=_extract_entities(request.question, ""),
                            top_k=max(1, len(ids)), max_hops=self.max_hops), set(ids)

    def _safe_history(self, request: WorkloadRequest, hits: list[EvidenceHit], current_ids: set[str]):
        query = _tokens(request.question)
        stop = {"a", "an", "the", "of", "in", "at", "on", "is", "was", "were", "are", "and",
                "to", "for", "who", "what", "which", "when", "where", "did", "does", "do"}
        query -= stop
        accepted = []
        with self.graph._lock:
            nodes = dict(self.graph.nodes)
        summarized_source_ids = {
            source_id
            for node in nodes.values()
            if node.metadata.get("memory_representation") and node.metadata.get("source_only_memory")
            for source_id in node.metadata.get("provenance_sentence_ids", ())
        }
        for hit in hits:
            if hit.node_id in current_ids:
                continue
            # Once a source-only abstractive memory covers an older sentence,
            # retrieve the compact note instead of also re-inserting that raw
            # historical sentence. The raw node remains in the graph as its
            # provenance anchor and can be inspected independently.
            if hit.node_id in summarized_source_ids:
                continue
            node = nodes.get(hit.node_id)
            if node is None:
                continue
            if node.metadata.get("abstractive_summary_pending"):
                continue
            if node.metadata.get("memory_representation") and not node.metadata.get("source_only_memory"):
                continue
            terms = _tokens(hit.title + " " + hit.text) - stop
            if len(query & terms) < min(2, len(query)) or not query:
                continue
            accepted.append(hit)
        return accepted[:self.top_k_candidates]

    def _pack_fused(self, request: WorkloadRequest, current: list[EvidenceHit],
                    historical: list[EvidenceHit], tokenizer: Tokenizer, budget: int):
        prefix = "Answer the question using only the retrieved context.\nContext:\n"
        suffix = f"\nQuestion: {request.question}\nAnswer:"
        fixed = tokenizer.count_tokens(prefix + suffix)
        effective_budget = max(budget, fixed)
        available = max(0, effective_budget - fixed)
        current_ids = {h.node_id for h in current}
        unique = {h.node_id: h for h in [*current, *historical]}
        hits = list(unique.values())
        candidates = [(h.title, h.source_sentence_index or 0, h.text,
                       _tokens(h.title + " " + h.text)) for h in hits]
        scores = self._ranker._bm25_scores(candidates, request.question)
        costs = [tokenizer.count_tokens((f"[{h.title}] " if h.title else "") + h.text + "\n") for h in hits]
        selected: list[int] = []
        seen_text: set[str] = set()
        doc_counts: Counter = Counter()
        used = history_used = 0
        remaining = set(range(len(hits)))
        history_limit = int(available * self.max_history_fraction) if current else available
        while remaining:
            feasible = []
            for i in sorted(remaining):
                hit = hits[i]
                is_history = hit.node_id not in current_ids
                if used + costs[i] > available or (is_history and history_used + costs[i] > history_limit):
                    continue
                if hit.text.casefold().strip() in seen_text:
                    continue
                # Relevance with a document-diversity bonus preserves bridge
                # material without requiring gold supporting-fact labels.
                redundancy = max((self._ranker._jaccard(candidates[i][3], candidates[j][3])
                                  for j in selected), default=0.0)
                doc = hit.source_document_id or hit.title
                priority = (0.80 * scores[i] - 0.12 * redundancy
                            + 0.14 / (1 + doc_counts[doc])
                            + (0.08 if not is_history else 0.0))
                feasible.append((priority, -costs[i], -i, i))
            if not feasible:
                break
            i = max(feasible)[-1]
            remaining.remove(i)
            selected.append(i)
            used += costs[i]
            if hits[i].node_id not in current_ids:
                history_used += costs[i]
            seen_text.add(hits[i].text.casefold().strip())
            doc_counts[hits[i].source_document_id or hits[i].title] += 1
        selected_hits = sorted((hits[i] for i in selected),
                               key=lambda h: (h.source_document_id or "", h.source_sentence_index or 0, h.node_id))

        def render(items):
            return prefix + "\n".join((f"[{h.title}] " if h.title else "") + h.text for h in items) + suffix

        prompt = render(selected_hits)
        while selected_hits and tokenizer.count_tokens(prompt) > effective_budget:
            selected_hits.pop()
            prompt = render(selected_hits)
        # If a single sentence exceeds the budget, trim only the evidence;
        # never cut off the question or the answer delimiter.
        if not selected_hits and available and current:
            best = max(range(len(current)), key=lambda i: (scores[i], -costs[i], -i))
            hit = current[best]
            text_budget = max(0, available - tokenizer.count_tokens(f"[{hit.title}] \n"))
            text = tokenizer.truncate_to_tokens(hit.text, text_budget)
            if text:
                selected_hits = [replace(hit, text=text)]
                prompt = render(selected_hits)
                while text and tokenizer.count_tokens(prompt) > effective_budget:
                    text_budget -= 1
                    text = tokenizer.truncate_to_tokens(hit.text, max(0, text_budget))
                    selected_hits = [replace(hit, text=text)] if text else []
                    prompt = render(selected_hits)
        return prompt, selected_hits, effective_budget

    def compress_request(self, request: WorkloadRequest, *, tokenizer: Tokenizer,
                         target_budget_ratio: float | None = None) -> CompressionRecord:
        with measure_ms() as measured:
            original_tokens = tokenizer.count_tokens(request.prompt)
            ratio = self.retain_ratio if target_budget_ratio is None else target_budget_ratio
            _validate_ratio(ratio)
            fingerprint = sha256((str(request.metadata.get("dataset", "")) + "\0" + request.context).encode()).hexdigest()
            with self._state_lock:
                cached_ids = self._cache.get(fingerprint)
            with self.graph._lock:
                before_nodes = len(self.graph.nodes)
            history_future = self._retrieval_executor.submit(
                self.graph.search, request.question, query_entities=_extract_entities(request.question, ""),
                top_k=max(1, before_nodes), max_hops=self.max_hops)
            current_future = self._retrieval_executor.submit(self._search_current, request)
            done, _ = wait((history_future, current_future), return_when=FIRST_COMPLETED)
            historical_hits = history_future.result()
            early = cached_ids is not None and history_future in done
            if early:
                current_ids = set(cached_ids)
                current_hits = [h for h in historical_hits if h.node_id in current_ids]
                early = {h.node_id for h in current_hits} == current_ids and bool(current_ids)
            if not early:
                current_hits, current_ids = current_future.result()
            historical_hits = self._safe_history(request, historical_hits, current_ids)
            prompt, selected_hits, budget = self._pack_fused(
                request, current_hits, historical_hits, tokenizer, max(1, int(original_tokens * ratio)))
            # Keep immutable source anchors, including unselected material,
            # for the post-generation reconciliation and provenance checks.
            ingest_request_context(request, graph=self.graph, status="uncertain")
            with self._state_lock:
                self._cache[fingerprint] = current_ids
                self._retrieval_futures.append(current_future)
            selected_ids = {h.node_id for h in selected_hits}
            history_selected = [h for h in selected_hits if h.node_id not in current_ids]
        return CompressionRecord(
            method=self.name, original_prompt_tokens=original_tokens,
            compressed_prompt_tokens=tokenizer.count_tokens(prompt), compressed_prompt=prompt,
            latency_ms=measured[0], target_budget_ratio=target_budget_ratio,
            metadata={
                "graph_nodes_before": before_nodes, "graph_nodes_after": len(self.graph.nodes),
                "candidate_evidence_count": len(current_hits) + len(historical_hits),
                "selected_evidence_count": len(selected_hits),
                "selected_sentence_ids": [h.node_id for h in selected_hits],
                "current_sentence_ids": sorted(current_ids),
                "reconciliation_sentence_ids": sorted(current_ids | {h.node_id for h in historical_hits}),
                "historical_candidate_count": len(historical_hits),
                "reused_evidence_count": len(history_selected),
                "compressed_memory_selected_count": sum(h.source_document_id is not None and
                    h.source_document_id.startswith("history:") for h in history_selected),
                "evidence_reuse_rate": len(history_selected) / max(1, len(selected_ids)),
                "budget_tokens": budget, "requested_budget_tokens": int(original_tokens * ratio),
                "dynamic_budget_ratio": ratio,
                "retrieval_audit": {"winner": "session_graph" if early else "fused",
                    "history_shortcut": early, "source_coverage_complete": cached_ids is not None,
                    "current_candidate_count": len(current_hits),
                    "history_candidate_count": len(historical_hits),
                    "background_pending": early and not current_future.done()},
            })

    def wait_for_retrieval(self, timeout: float | None = None) -> None:
        with self._state_lock:
            futures, self._retrieval_futures = self._retrieval_futures, []
        for future in futures:
            future.result(timeout=timeout)

    def close(self) -> None:
        self.wait_for_retrieval()
        self._retrieval_executor.shutdown(wait=True)
        super().close()
