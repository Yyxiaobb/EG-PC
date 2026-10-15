"""Session-scoped evidence-graph compression."""

from __future__ import annotations

from threading import RLock
from typing import Any

from ladc.compressors.dual_path import DualPathEvidenceGraphCompressor
from ladc.evidence_graph import EvidenceGraph
from ladc.interfaces import Tokenizer
from ladc.records import CompressionRecord
from ladc.workloads import WorkloadRequest


def request_session_id(request: WorkloadRequest) -> str:
    """Resolve the application conversation/question-family identifier."""

    metadata = request.metadata
    for key in ("session_id", "question_family_id", "conversation_id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "default"


class SessionEvidenceGraphCompressor:
    """Keep one graph and compressor instance per application session."""

    name = "evidence_graph_session"

    def __init__(self, *, compressor_kwargs: dict[str, Any] | None = None) -> None:
        self._compressor_kwargs = dict(compressor_kwargs or {})
        self._compressors: dict[str, DualPathEvidenceGraphCompressor] = {}
        self._lock = RLock()

    def _for_session(self, session_id: str) -> EvidenceGraphCompressor:
        with self._lock:
            compressor = self._compressors.get(session_id)
            if compressor is None:
                compressor = DualPathEvidenceGraphCompressor(
                    graph=EvidenceGraph(),
                    **self._compressor_kwargs,
                )
                self._compressors[session_id] = compressor
            return compressor

    def compress_request(
        self,
        request: WorkloadRequest,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        return self._for_session(request_session_id(request)).compress_request(
            request,
            tokenizer=tokenizer,
            target_budget_ratio=target_budget_ratio,
        )

    def compress(
        self,
        prompt: str,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        # Prompt-only callers have no session metadata and use the default graph.
        return self._for_session("default").compress(
            prompt,
            tokenizer=tokenizer,
            target_budget_ratio=target_budget_ratio,
        )

    def record_generation(
        self,
        request: WorkloadRequest,
        output_text: str,
        *,
        compression: CompressionRecord,
        tokenizer: Tokenizer,
    ) -> None:
        self._for_session(request_session_id(request)).record_generation(
            request,
            output_text,
            compression=compression,
            tokenizer=tokenizer,
        )

    def session_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._compressors)

    def graph_for_session(self, session_id: str) -> EvidenceGraph:
        return self._for_session(session_id).graph

    def wait_for_history(self, timeout: float | None = None) -> None:
        with self._lock:
            compressors = tuple(self._compressors.values())
        for compressor in compressors:
            compressor.wait_for_history(timeout=timeout)

    def close(self) -> None:
        with self._lock:
            compressors = tuple(self._compressors.values())
        for compressor in compressors:
            compressor.close()
