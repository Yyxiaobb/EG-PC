"""Latency-aware dynamic prompt compression."""

from ladc.evidence_builder import build_evidence_graph, ingest_request_context
from ladc.evidence_graph import EvidenceGraph, EvidenceHit, EvidenceSpan

__version__ = "0.1.0"

__all__ = [
    "EvidenceGraph",
    "EvidenceHit",
    "EvidenceSpan",
    "build_evidence_graph",
    "ingest_request_context",
    "__version__",
]
