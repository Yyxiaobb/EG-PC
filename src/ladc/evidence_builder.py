"""Runtime adapters from benchmark workloads to the evidence graph."""

from __future__ import annotations

from hashlib import sha1
import re
from typing import Iterable

from ladc.evidence_graph import EvidenceGraph, EvidenceSpan, EvidenceStatus, _stable_id
from ladc.workloads import WorkloadRequest


_BLOCK_HEADER = re.compile(r"^\[([^\]\n]+)\]\s*(.*)$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")
_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’-]*)(?:\s+[A-Z][A-Za-z0-9'’-]*)*\b"
)
_ENTITY_STOP_WORDS = {
    "Answer",
    "Context",
    "Document",
    "Question",
    "The",
    "This",
    "That",
    "Who",
    "Which",
    "What",
    "When",
    "Where",
    "Why",
    "How",
}


def _stable_document_id(dataset: str, title: str, text: str) -> str:
    digest = sha1(text.strip().encode("utf-8")).hexdigest()[:16]
    return f"{dataset}:{title.strip()}:{digest}"


def _split_blocks(context: str, titles: Iterable[str] = ()) -> list[tuple[str, str]]:
    """Parse the normalized ``[title] text`` context format.

    Dataset loaders currently emit one titled block per line.  The fallback
    keeps raw contexts usable when a custom dataset has no title markers.
    """

    lines = [line.strip() for line in context.splitlines() if line.strip()]
    blocks: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in lines:
        match = _BLOCK_HEADER.match(line)
        if match:
            if current_lines:
                blocks.append((current_title, " ".join(current_lines).strip()))
            current_title = match.group(1).strip()
            current_lines = [match.group(2).strip()] if match.group(2).strip() else []
        else:
            current_lines.append(line)
    if current_lines:
        blocks.append((current_title, " ".join(current_lines).strip()))

    if blocks:
        return [(title or f"Document {index}", text) for index, (title, text) in enumerate(blocks) if text]
    raw = context.strip()
    if not raw:
        return []
    title_list = list(titles)
    return [(title_list[0] if title_list else "Document 0", raw)]


def _split_sentences(text: str) -> list[str]:
    pieces = [piece.strip() for piece in _SENTENCE_BOUNDARY.split(text) if piece.strip()]
    return pieces or ([text.strip()] if text.strip() else [])


def _extract_entities(text: str, title: str) -> tuple[str, ...]:
    candidates = [title, *_ENTITY_PATTERN.findall(text)]
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = " ".join(candidate.split()).strip(" ,.;:()[]")
        if not normalized or normalized in _ENTITY_STOP_WORDS or normalized.casefold() in seen:
            continue
        seen.add(normalized.casefold())
        unique.append(normalized)
    return tuple(unique)


def ingest_request_context(
    request: WorkloadRequest,
    *,
    graph: EvidenceGraph | None = None,
    turn_id: str | None = None,
    status: EvidenceStatus = "uncertain",
    confidence: float = 0.7,
) -> tuple[EvidenceGraph, tuple[str, ...]]:
    """Add runtime-visible context from one request to an evidence graph.

    Gold ``references`` and ``supporting_facts`` are intentionally ignored.
    The returned sentence IDs can be used by a later compressor to mark the
    actually selected evidence after generation or verification.
    """

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    graph = graph or EvidenceGraph()
    dataset = str(request.metadata.get("dataset", "unknown"))
    titles = request.metadata.get("context_titles", ())
    if not isinstance(titles, (list, tuple)):
        titles = ()
    spans: list[EvidenceSpan] = []
    for title, block in _split_blocks(request.context, titles):
        document_id = _stable_document_id(dataset, title, block)
        for sentence_index, sentence in enumerate(_split_sentences(block)):
            spans.append(
                EvidenceSpan(
                    document_id=document_id,
                    sentence_index=sentence_index,
                    text=sentence,
                    title=title,
                    entities=_extract_entities(sentence, title),
                    confidence=confidence,
                    status=status,
                )
            )
    graph.add_turn(turn_id or request.request_id, request.question, spans)
    # A repeated source retains its first turn_id when nodes merge. Identify
    # current sources by immutable provenance, not by their insertion turn.
    sentence_ids = tuple(_stable_id("sentence", span.document_id, span.sentence_index, span.text)
                         for span in spans)
    return graph, sentence_ids


def build_evidence_graph(
    requests: Iterable[WorkloadRequest],
    *,
    graph: EvidenceGraph | None = None,
    status: EvidenceStatus = "uncertain",
    confidence: float = 0.7,
) -> EvidenceGraph:
    """Build a graph from a sequence of requests without using annotations."""

    graph = graph or EvidenceGraph()
    for request in requests:
        ingest_request_context(
            request,
            graph=graph,
            status=status,
            confidence=confidence,
        )
    return graph
