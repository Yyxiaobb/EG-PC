"""A small provenance-aware evidence graph for multi-turn compression.

The graph is deliberately dependency-free.  It stores source sentences rather
than trusting generated answers as facts, which makes it suitable for the
runtime path of a latency experiment and for later offline inspection.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from hashlib import sha1
import json
from pathlib import Path
import re
from threading import RLock
from typing import Any, Iterable, Literal, Mapping


NodeType = Literal["question", "document", "sentence", "entity", "claim", "answer"]
EvidenceStatus = Literal["verified", "uncertain", "conflicted", "expired"]


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "which",
    "who",
    "where",
    "when",
    "with",
}


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    digest = sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(value.casefold())
        if token not in _STOP_WORDS
    }


@dataclass(frozen=True)
class EvidenceSpan:
    """A source-backed sentence that may be written to the graph."""

    document_id: str
    sentence_index: int
    text: str
    title: str = ""
    entities: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()
    confidence: float = 1.0
    status: EvidenceStatus = "verified"

    def __post_init__(self) -> None:
        if self.sentence_index < 0:
            raise ValueError("sentence_index must be non-negative")
        if not self.text.strip():
            raise ValueError("evidence text must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    node_type: NodeType
    text: str
    source_document_id: str | None = None
    source_sentence_index: int | None = None
    turn_id: str | None = None
    confidence: float = 1.0
    status: EvidenceStatus = "verified"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceHit:
    """A ranked source sentence returned to a downstream compressor."""

    node_id: str
    text: str
    score: float
    hops: int
    path: tuple[str, ...]
    source_document_id: str | None
    source_sentence_index: int | None
    title: str
    confidence: float
    status: EvidenceStatus
    turn_id: str | None


class EvidenceGraph:
    """In-memory provenance graph with deterministic JSON persistence.

    The graph uses lightweight lexical matching for its first implementation.
    A later compressor can replace ``search`` with BM25 or embeddings without
    changing the stored evidence contract.
    """

    schema_version = 1

    def __init__(self) -> None:
        self._lock = RLock()
        self.nodes: dict[str, EvidenceNode] = {}
        self.edges: dict[tuple[str, str, str], EvidenceEdge] = {}
        self._adjacency: dict[str, set[str]] = defaultdict(set)

    def add_compressed_turn(
        self,
        turn_id: str,
        question: str,
        answer: str,
        evidence: Iterable[EvidenceSpan],
        *,
        tokenizer: Any | None = None,
        question_tokens: int = 48,
        answer_tokens: int = 96,
        memory_text: str | None = None,
        memory_provenance: Iterable[str] = (),
    ) -> str:
        """Store a post-generation, memory-sized interaction.

        Evidence is already compressed by selection: only the sentences used
        for the generated prompt are retained.  The question and answer are
        additionally bounded before being added as contextual nodes.  Source
        spans keep their original provenance and are never replaced by the
        generated answer.
        """

        def bound(text: str, limit: int) -> str:
            text = text.strip()
            if not text or limit <= 0:
                return ""
            if tokenizer is not None:
                return tokenizer.truncate_to_tokens(text, limit).strip()
            words = text.split()
            return " ".join(words[:limit])

        compact_question = bound(question, question_tokens)
        compact_answer = bound(answer, answer_tokens)
        with self._lock:
            question_id = self.add_turn(
                turn_id,
                compact_question,
                tuple(evidence),
                answer=compact_answer,
                answer_confidence=0.7,
            )
            if memory_text and memory_text.strip():
                memory_span_id = self.add_span(
                    EvidenceSpan(
                        document_id=f"history:{turn_id}",
                        sentence_index=0,
                        text=memory_text.strip(),
                        title="compressed history",
                        confidence=0.7,
                        status="uncertain",
                    ),
                    turn_id=turn_id,
                )
                memory_node = self.nodes.get(memory_span_id)
                if memory_node is not None:
                    self.nodes[memory_span_id] = EvidenceNode(
                        node_id=memory_node.node_id,
                        node_type=memory_node.node_type,
                        text=memory_node.text,
                        source_document_id=memory_node.source_document_id,
                        source_sentence_index=memory_node.source_sentence_index,
                        turn_id=memory_node.turn_id,
                        confidence=memory_node.confidence,
                        status=memory_node.status,
                        metadata={
                            **memory_node.metadata,
                            "memory_representation": True,
                            "provenance_sentence_ids": list(memory_provenance),
                        },
                    )
                self.add_edge(question_id, memory_span_id, "compressed_memory")
            return question_id

    def add_node(self, node: EvidenceNode) -> str:
        """Insert or merge a node while preserving the stronger metadata."""

        if node.node_id in self.nodes:
            existing = self.nodes[node.node_id]
            merged_metadata = {**existing.metadata, **node.metadata}
            merged = EvidenceNode(
                node_id=existing.node_id,
                node_type=existing.node_type,
                text=existing.text or node.text,
                source_document_id=existing.source_document_id or node.source_document_id,
                source_sentence_index=(
                    existing.source_sentence_index
                    if existing.source_sentence_index is not None
                    else node.source_sentence_index
                ),
                turn_id=existing.turn_id or node.turn_id,
                confidence=max(existing.confidence, node.confidence),
                status=existing.status if existing.status == "verified" else node.status,
                metadata=merged_metadata,
            )
            self.nodes[node.node_id] = merged
            return node.node_id
        self.nodes[node.node_id] = node
        return node.node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Add a directed relation and expose it to undirected graph search."""

        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError("both edge endpoints must already exist in the graph")
        edge = EvidenceEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            metadata=dict(metadata or {}),
        )
        self.edges[(source_id, target_id, relation)] = edge
        self._adjacency[source_id].add(target_id)
        self._adjacency[target_id].add(source_id)
        return f"{source_id}->{relation}->{target_id}"

    def add_turn(
        self,
        turn_id: str,
        question: str,
        evidence: Iterable[EvidenceSpan],
        *,
        answer: str | None = None,
        answer_confidence: float = 0.0,
    ) -> str:
        """Add one interaction and its source-backed evidence spans.

        ``answer`` is stored only as a contextual node.  Search ranks sentence
        nodes, never answer nodes, so a generated answer cannot become an
        unsupported fact by itself.
        """

        with self._lock:
            if not turn_id.strip():
                raise ValueError("turn_id must not be empty")
            question_id = _stable_id("question", turn_id, question)
            self.add_node(
                EvidenceNode(
                    node_id=question_id,
                    node_type="question",
                    text=question,
                    turn_id=turn_id,
                )
            )
            answer_id: str | None = None
            if answer is not None:
                answer_id = _stable_id("answer", turn_id, answer)
                self.add_node(
                    EvidenceNode(
                        node_id=answer_id,
                        node_type="answer",
                        text=answer,
                        turn_id=turn_id,
                        confidence=answer_confidence,
                        status="uncertain" if answer_confidence < 0.8 else "verified",
                    )
                )
                self.add_edge(question_id, answer_id, "answered_by")

            for span in evidence:
                sentence_id = self.add_span(span, turn_id=turn_id)
                self.add_edge(question_id, sentence_id, "retrieved")
                if answer_id is not None:
                    self.add_edge(sentence_id, answer_id, "supports")
            return question_id

    def add_span(self, span: EvidenceSpan, *, turn_id: str | None = None) -> str:
        """Add a source sentence plus document, entity and claim relations."""

        document_id = _stable_id("document", span.document_id)
        sentence_id = _stable_id(
            "sentence",
            span.document_id,
            span.sentence_index,
            span.text,
        )
        self.add_node(
            EvidenceNode(
                node_id=document_id,
                node_type="document",
                text=span.title or span.document_id,
                source_document_id=span.document_id,
                metadata={"title": span.title},
            )
        )
        self.add_node(
            EvidenceNode(
                node_id=sentence_id,
                node_type="sentence",
                text=span.text,
                source_document_id=span.document_id,
                source_sentence_index=span.sentence_index,
                turn_id=turn_id,
                confidence=span.confidence,
                status=span.status,
                metadata={"title": span.title},
            )
        )
        self.add_edge(sentence_id, document_id, "located_in")

        for entity in span.entities:
            normalized = _normalize(entity)
            if not normalized:
                continue
            entity_id = _stable_id("entity", normalized)
            self.add_node(EvidenceNode(node_id=entity_id, node_type="entity", text=entity))
            self.add_edge(sentence_id, entity_id, "mentions")
        for claim in span.claims:
            normalized = _normalize(claim)
            if not normalized:
                continue
            claim_id = _stable_id("claim", normalized)
            self.add_node(
                EvidenceNode(
                    node_id=claim_id,
                    node_type="claim",
                    text=claim,
                    source_document_id=span.document_id,
                    source_sentence_index=span.sentence_index,
                    turn_id=turn_id,
                    confidence=span.confidence,
                    status=span.status,
                )
            )
            self.add_edge(claim_id, sentence_id, "supported_by")
        return sentence_id

    def search(
        self,
        query: str,
        *,
        query_entities: Iterable[str] = (),
        top_k: int = 8,
        max_hops: int = 2,
        allowed_statuses: Iterable[EvidenceStatus] = ("verified", "uncertain"),
    ) -> list[EvidenceHit]:
        """Find source sentences using lexical relevance and graph bridges.

        A sentence can be returned because it directly matches the query or
        because it is connected through a query entity.  The latter is the
        first implementation of evidence-graph expansion: a bridge sentence
        can survive even when its wording has little direct query overlap.
        """

        if top_k <= 0:
            return []
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        with self._lock:
            node_values = tuple(self.nodes.values())
            node_map = dict(self.nodes)
            adjacency = {node_id: tuple(neighbors) for node_id, neighbors in self._adjacency.items()}
        query_terms = _tokens(query)
        normalized_entities = {_normalize(entity) for entity in query_entities if _normalize(entity)}
        entity_ids = {
            node.node_id
            for node in node_values
            if node.node_type == "entity"
            and (
                _normalize(node.text) in normalized_entities
                or _normalize(node.text) in _tokens(query)
                or _normalize(node.text) in _normalize(query)
            )
        }

        distances: dict[str, int] = {}
        paths: dict[str, tuple[str, ...]] = {}
        queue: deque[str] = deque()
        for node_id in entity_ids:
            distances[node_id] = 0
            paths[node_id] = (node_id,)
            queue.append(node_id)
        while queue:
            current = queue.popleft()
            distance = distances[current]
            if distance >= max_hops:
                continue
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in distances:
                    continue
                distances[neighbor] = distance + 1
                paths[neighbor] = (*paths[current], neighbor)
                queue.append(neighbor)

        allowed = set(allowed_statuses)
        hits: list[EvidenceHit] = []
        for node in node_values:
            if node.node_type != "sentence" or node.status not in allowed:
                continue
            sentence_terms = _tokens(node.text)
            lexical = len(query_terms & sentence_terms) / max(1, len(query_terms))
            graph_distance = distances.get(node.node_id)
            bridge_bonus = 0.0
            path = (node.node_id,)
            hops = max_hops + 1
            if graph_distance is not None:
                bridge_bonus = 0.35 / (1 + graph_distance)
                path = (*paths[node.node_id],) if node.node_id in paths else path
                hops = graph_distance
            entity_bonus = 0.0
            if normalized_entities:
                mentioned = {
                    _normalize(node_map[neighbor].text)
                    for neighbor in adjacency.get(node.node_id, ())
                    if neighbor in node_map and node_map[neighbor].node_type == "entity"
                }
                entity_bonus = 0.35 * len(mentioned & normalized_entities) / len(normalized_entities)
            score = lexical + entity_bonus + bridge_bonus + 0.1 * node.confidence
            if score <= 0.0:
                continue
            title = str(node.metadata.get("title", ""))
            hits.append(
                EvidenceHit(
                    node_id=node.node_id,
                    text=node.text,
                    score=score,
                    hops=hops,
                    path=path,
                    source_document_id=node.source_document_id,
                    source_sentence_index=node.source_sentence_index,
                    title=title,
                    confidence=node.confidence,
                    status=node.status,
                    turn_id=node.turn_id,
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.hops, hit.source_document_id or "", hit.source_sentence_index or 0))
        return hits[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges.values()],
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceGraph":
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError(f"unsupported evidence graph schema: {payload.get('schema_version')}")
        graph = cls()
        for raw_node in payload.get("nodes", []):
            graph.add_node(EvidenceNode(**raw_node))
        for raw_edge in payload.get("edges", []):
            graph.add_edge(
                raw_edge["source_id"],
                raw_edge["target_id"],
                raw_edge["relation"],
                weight=raw_edge.get("weight", 1.0),
                metadata=raw_edge.get("metadata", {}),
            )
        return graph

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("evidence graph file must contain a JSON object")
        return cls.from_dict(payload)
