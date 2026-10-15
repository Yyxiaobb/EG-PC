"""Leakage-safe construction of related and unrelated QA sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ladc.evidence_builder import _extract_entities, _split_blocks
from ladc.evidence_graph import _normalize


@dataclass(frozen=True)
class SequenceSpec:
    sequence_id: str
    dataset: str
    kind: str
    requests: tuple[Any, ...]
    adjacent_relations: tuple[dict[str, list[str]], ...]


def _titles(request: Any) -> set[str]:
    return {
        _normalize(str(title))
        for title in request.metadata.get("context_titles", ())
        if str(title).strip()
    }


def _query_entities(request: Any) -> set[str]:
    return {
        _normalize(entity)
        for entity in _extract_entities(request.question, "")
        if _normalize(entity)
    }


def _context_entities(request: Any) -> set[str]:
    entities: set[str] = set()
    context_titles = _titles(request)
    for title, block in _split_blocks(request.context, request.metadata.get("context_titles", ())):
        # ``_split_blocks`` uses ``Document N`` for an untitled raw block;
        # that synthetic label must not become a cross-request relation.
        entity_title = "" if title.casefold().startswith("document ") else title
        for entity in _extract_entities(block, entity_title):
            normalized = _normalize(entity)
            if not normalized:
                continue
            raw = entity.strip()
            # Sentence-initial single capitalized words (e.g. "Although",
            # "He", or "In") are not reliable cross-request entities.
            is_multiword = len(normalized.split()) >= 2
            is_acronym = len(raw) >= 2 and raw.isupper()
            if normalized in context_titles or is_multiword or is_acronym:
                entities.add(normalized)
    return {entity for entity in entities if entity}


def relation_features(left: Any, right: Any) -> dict[str, list[str]]:
    """Return runtime-visible links between two requests.

    Gold answers and supporting-fact annotations are intentionally not used.
    """

    shared_titles = sorted(_titles(left) & _titles(right))
    shared_query_entities = sorted(_query_entities(left) & _query_entities(right))
    shared_context_entities = sorted(_context_entities(left) & _context_entities(right))
    return {
        "shared_titles": shared_titles,
        "shared_query_entities": shared_query_entities,
        "shared_context_entities": shared_context_entities,
    }


def _is_related(left: Any, right: Any) -> bool:
    features = relation_features(left, right)
    return bool(
        features["shared_titles"]
        or features["shared_query_entities"]
        or features["shared_context_entities"]
    )


def build_related_sequence_specs(
    requests_by_dataset: dict[str, list[Any]],
    *,
    sequence_length: int = 3,
    max_sequences_per_dataset: int = 2,
) -> list[SequenceSpec]:
    """Build chains where every adjacent pair shares a visible link."""

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    specs: list[SequenceSpec] = []
    for dataset, source in requests_by_dataset.items():
        unused = [request for request in source if request.references and _titles(request)]
        unused.sort(key=lambda request: request.request_id)
        sequence_number = 0
        while unused and sequence_number < max_sequences_per_dataset:
            seed = unused.pop(0)
            sequence = [seed]
            relations: list[dict[str, list[str]]] = []
            while len(sequence) < sequence_length:
                candidates = [request for request in unused if _is_related(sequence[-1], request)]
                if not candidates:
                    break
                candidates.sort(
                    key=lambda request: (
                        -len(relation_features(sequence[-1], request)["shared_titles"]),
                        -len(relation_features(sequence[-1], request)["shared_query_entities"]),
                        -len(relation_features(sequence[-1], request)["shared_context_entities"]),
                        request.request_id,
                    )
                )
                next_request = candidates[0]
                unused.remove(next_request)
                relations.append(relation_features(sequence[-1], next_request))
                sequence.append(next_request)
            if len(sequence) == sequence_length:
                specs.append(
                    SequenceSpec(
                        sequence_id=f"{dataset}_related_{sequence_number}",
                        dataset=dataset,
                        kind="related",
                        requests=tuple(sequence),
                        adjacent_relations=tuple(relations),
                    )
                )
                sequence_number += 1
            else:
                unused[0:0] = sequence[1:]
                break
    return specs


def build_unrelated_sequence_specs(
    requests_by_dataset: dict[str, list[Any]],
    *,
    sequence_length: int = 3,
    max_sequences_per_dataset: int = 1,
) -> list[SequenceSpec]:
    """Build negative-control chains with no title/entity links."""

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    specs: list[SequenceSpec] = []
    for dataset, source in requests_by_dataset.items():
        candidates = [request for request in source if request.references and _titles(request)]
        candidates.sort(key=lambda request: request.request_id)
        used: set[str] = set()
        for sequence_number in range(max_sequences_per_dataset):
            sequence: list[Any] = []
            for request in candidates:
                if request.request_id in used:
                    continue
                if all(not _is_related(existing, request) for existing in sequence):
                    sequence.append(request)
                    used.add(request.request_id)
                if len(sequence) == sequence_length:
                    break
            if len(sequence) != sequence_length:
                break
            specs.append(
                SequenceSpec(
                    sequence_id=f"{dataset}_unrelated_{sequence_number}",
                    dataset=dataset,
                    kind="unrelated",
                    requests=tuple(sequence),
                    adjacent_relations=tuple(relation_features(left, right) for left, right in zip(sequence, sequence[1:])),
                )
            )
    return specs
