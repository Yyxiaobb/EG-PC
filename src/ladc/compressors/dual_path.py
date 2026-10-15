"""Parallel current-material and session-memory evidence retrieval."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Lock
from typing import Any

from ladc.compressors.evidence_graph import EvidenceGraphCompressor, _extract_entities
from ladc.evidence_builder import ingest_request_context
from ladc.evidence_graph import EvidenceGraph, EvidenceHit
from ladc.interfaces import Tokenizer
from ladc.records import CompressionRecord
from ladc.timing import measure_ms
from ladc.workloads import WorkloadRequest


class DualPathEvidenceGraphCompressor(EvidenceGraphCompressor):
    """Race historical-memory and current-context retrieval.

    The first branch that produces a minimally usable candidate supplies the
    prompt. The other branch continues in the background; current context is
    ingested into the session graph so its evidence is available to later
    turns even when it did not win the race.
    """

    name = "evidence_graph_dual_path"

    def __init__(self, graph: EvidenceGraph | None = None, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init__(graph=graph, name=name or self.name, **kwargs)
        self._retrieval_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ladc-retrieval")
        self._retrieval_futures: set[Future[Any]] = set()
        self._retrieval_futures_lock = Lock()
        self._audits: dict[str, dict[str, Any]] = {}
        self._audits_lock = Lock()

    def _track_retrieval(self, future: Future[Any]) -> Future[Any]:
        with self._retrieval_futures_lock:
            self._retrieval_futures.add(future)
        future.add_done_callback(self._forget_retrieval)
        return future

    def _forget_retrieval(self, future: Future[Any]) -> None:
        with self._retrieval_futures_lock:
            self._retrieval_futures.discard(future)

    def _search_current(self, request: WorkloadRequest) -> list[EvidenceHit]:
        graph = EvidenceGraph()
        _, sentence_ids = ingest_request_context(
            request,
            graph=graph,
            turn_id=request.request_id,
            status="uncertain",
        )
        return graph.search(
            request.question,
            query_entities=_extract_entities(request.question, ""),
            top_k=self.top_k_candidates,
            max_hops=self.max_hops,
        )

    def _finish_background_audit(
        self,
        request_id: str,
        branch: str,
        future: Future[list[EvidenceHit]],
    ) -> None:
        try:
            hits = future.result()
            error = ""
        except Exception as exc:  # pragma: no cover - defensive worker path
            hits = []
            error = type(exc).__name__
        with self._audits_lock:
            audit = self._audits.get(request_id)
            if audit is None:
                return
            audit.update(
                {
                    "background_branch": branch,
                    "background_pending": False,
                    "background_candidate_count": len(hits),
                    "background_best_score": hits[0].score if hits else 0.0,
                    "background_error": error,
                }
            )

    @staticmethod
    def _ready(hits: list[EvidenceHit]) -> bool:
        return bool(hits) and hits[0].score >= 0.12

    def compress_request(
        self,
        request: WorkloadRequest,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        original_tokens = tokenizer.count_tokens(request.prompt)
        with measure_ms() as measured:
            before_nodes = len(self.graph.nodes)
            historical_future = self._track_retrieval(
                self._retrieval_executor.submit(
                    self.graph.search,
                    request.question,
                    query_entities=_extract_entities(request.question, ""),
                    top_k=self.top_k_candidates,
                    max_hops=self.max_hops,
                )
            )
            current_future = self._track_retrieval(
                self._retrieval_executor.submit(self._search_current, request)
            )
            done, pending = wait((historical_future, current_future), return_when=FIRST_COMPLETED)
            winner_name = ""
            winner_hits: list[EvidenceHit] = []
            for future, branch in ((historical_future, "session_graph"), (current_future, "current_material")):
                if future not in done:
                    continue
                try:
                    candidate = future.result()
                except Exception:
                    candidate = []
                if self._ready(candidate):
                    winner_name = branch
                    winner_hits = candidate
                    break
            if not winner_name:
                # Neither early result passed the gate; wait for the other
                # branch to prevent an under-supported answer.
                done_all, _ = wait((historical_future, current_future))
                pending = set()
                for future, branch in ((current_future, "current_material"), (historical_future, "session_graph")):
                    if future not in done_all:
                        continue
                    try:
                        candidate = future.result()
                    except Exception:
                        candidate = []
                    if self._ready(candidate) or candidate:
                        winner_name = branch
                        winner_hits = candidate
                        break

            # Persist all current raw evidence in the session graph, while
            # keeping the two retrieval paths logically separate.
            _, current_sentence_ids = ingest_request_context(
                request,
                graph=self.graph,
                turn_id=request.request_id,
                status="uncertain",
            )
            current_ids = set(current_sentence_ids)
            winner_hits, filter_metadata = self._filter_hits(request, winner_hits, current_ids)
            historical_hits = [hit for hit in winner_hits if hit.node_id not in current_ids]
            ratio = self._choose_ratio(
                target_budget_ratio,
                original_tokens=original_tokens,
                historical_hits=historical_hits,
                query=request.question,
            )
            budget = max(1, int(original_tokens * ratio))
            compressed_prompt, selected_hits, budget_used = self._pack(
                request,
                winner_hits,
                tokenizer=tokenizer,
                budget=budget,
                priority_ids=current_ids,
            )

        background_branch = "current_material" if winner_name == "session_graph" else "session_graph"
        audit = {
            "winner": winner_name or "none",
            "background_branch": background_branch,
            "background_pending": bool(pending),
            "background_future_ids": [id(future) for future in pending],
        }
        with self._audits_lock:
            self._audits[request.request_id] = audit
        for future, branch in ((historical_future, "session_graph"), (current_future, "current_material")):
            if future in pending:
                future.add_done_callback(
                    lambda completed, request_id=request.request_id, branch=branch: self._finish_background_audit(
                        request_id, branch, completed
                    )
                )
        selected_ids = {hit.node_id for hit in selected_hits}
        metadata = {
            "graph_nodes_before": before_nodes,
            "graph_nodes_after": len(self.graph.nodes),
            "candidate_evidence_count": len(winner_hits),
            "selected_evidence_count": len(selected_hits),
            "historical_candidate_count": len(historical_hits),
            "reused_evidence_count": len(selected_ids - current_ids),
            "evidence_reuse_rate": len(selected_ids - current_ids) / max(1, len(selected_hits)),
            "dynamic_budget_ratio": ratio,
            "budget_tokens": budget,
            "packed_evidence_tokens": budget_used,
            "selected_sentence_ids": [hit.node_id for hit in selected_hits],
            "retrieval_audit": audit,
        }
        metadata.update(filter_metadata)
        return CompressionRecord(
            method=self.name,
            original_prompt_tokens=original_tokens,
            compressed_prompt_tokens=tokenizer.count_tokens(compressed_prompt),
            compressed_prompt=compressed_prompt,
            latency_ms=measured[0],
            target_budget_ratio=target_budget_ratio,
            metadata=metadata,
        )

    def wait_for_retrieval(self, timeout: float | None = None) -> None:
        with self._retrieval_futures_lock:
            futures = tuple(self._retrieval_futures)
        for future in futures:
            future.result(timeout=timeout)

    def close(self) -> None:
        self.wait_for_retrieval()
        self._retrieval_executor.shutdown(wait=True)
        super().close()
