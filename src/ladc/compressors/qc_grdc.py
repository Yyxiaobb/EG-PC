"""Quality-constrained graph rate-distortion evidence compression.

QC-GRDC is a lightweight, training-free compressor for session evidence
graphs.  It treats a source sentence as a coded symbol with a token cost and
selects symbols by marginal information gain per token.  A protected quality
core covers query variables, graph bridges, and provenance anchors before the
remaining budget is spent on redundant context.

The quality signal is deliberately a runtime lower-bound proxy rather than a
claim about answer accuracy: every selected feature is observable from the
question and source graph, and every selected sentence retains its immutable
source identifier.  This makes the method usable without gold supporting
facts or an extra LLM call on the critical path.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Iterable

from ladc.compressors.evidence_graph import EvidenceGraphCompressor, _validate_ratio
from ladc.evidence_builder import _extract_entities, ingest_request_context
from ladc.evidence_graph import EvidenceGraph, EvidenceHit, _normalize, _tokens
from ladc.interfaces import Tokenizer
from ladc.records import CompressionRecord
from ladc.timing import measure_ms
from ladc.workloads import WorkloadRequest


_NUMBER_OR_DATE = re.compile(r"\b(?:\d{1,4}(?:[-/]\d{1,2}){0,2}|\d+(?:\.\d+)?)\b")
_NEGATIONS = {"no", "not", "never", "none", "without", "neither", "nor", "didn’t", "didn't"}
_QUESTION_WORDS = {
    "who", "what", "which", "when", "where", "why", "how", "whose", "whom",
}


def _feature_tokens(text: str) -> set[str]:
    """Return lexical answer-bearing features, including numbers and negation."""

    return _tokens(text) | {value.casefold() for value in _NUMBER_OR_DATE.findall(text)}


class QualityConstrainedGraphRateDistortionCompressor(EvidenceGraphCompressor):
    """Select graph evidence under a quality floor and a token budget.

    The selector has three stages:

    1. Estimate a full-reference feature set from all query-relevant graph
       hits.  Features include answer-bearing terms, query entities, source
       documents, and graph bridge nodes.
    2. Protect a quality core: shortest/strongest sentences that cover every
       high-weight query feature and at least one path for each bridge entity.
    3. Fill the remaining budget using marginal feature gain divided by native
       tokenizer cost (a rate-distortion objective).

    ``quality_floor`` is a conservative fraction of the full-reference proxy;
    it is not calculated from gold annotations.  Setting it to 1.0 requests
    all observable features that fit, while lower values trade coverage for
    latency.  Source IDs and paths are included in metadata for evaluation.
    """

    name = "evidence_graph_qc_grdc"

    def __init__(
        self,
        graph: EvidenceGraph | None = None,
        *,
        quality_floor: float = 0.92,
        min_retain_ratio: float = 0.30,
        max_retain_ratio: float = 0.80,
        top_k_candidates: int = 64,
        max_hops: int = 3,
        max_historical_fraction: float = 0.45,
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
        if not 0.0 < quality_floor <= 1.0:
            raise ValueError("quality_floor must be in the interval (0, 1]")
        if not 0.0 < max_historical_fraction <= 1.0:
            raise ValueError("max_historical_fraction must be in (0, 1]")
        self.quality_floor = quality_floor
        self.max_historical_fraction = max_historical_fraction

    @staticmethod
    def _query_features(question: str, hits: Iterable[EvidenceHit]) -> set[str]:
        query = _feature_tokens(question) - _QUESTION_WORDS
        # Query entities are kept as both the full entity and their component
        # tokens, so a bridge sentence can satisfy a multi-word entity.
        entities = set()
        for entity in _extract_entities(question, ""):
            normalized = _normalize(entity)
            if normalized:
                entities.add(normalized)
                entities.update(_feature_tokens(normalized))
        query.update(entities)
        return query

    @staticmethod
    def _candidate_features(hit: EvidenceHit, query_features: set[str]) -> set[str]:
        features = _feature_tokens(hit.text) & query_features
        if hit.title:
            features.update(_feature_tokens(hit.title) & query_features)
        # A source document feature discourages spending the entire budget on
        # near-duplicate sentences from one document.
        if hit.source_document_id:
            features.add("doc:" + hit.source_document_id)
        # Preserve the bridge structure.  The path is stable for a graph and
        # allows a downstream audit to verify that multi-hop support survived.
        for node_id in hit.path:
            features.add("path:" + node_id)
        features.add("sentence:" + hit.node_id)
        return features

    @staticmethod
    def _is_quality_feature(feature: str) -> bool:
        """Exclude bookkeeping IDs from the answer-quality proxy.

        Source and sentence IDs still participate in the selector, because
        they encode provenance and diversity, but counting every unique ID as
        answer quality would make the floor impossible to reach whenever the
        graph contains harmless duplicate passages.
        """

        if feature.startswith("doc:") or feature.startswith("sentence:"):
            return False
        if feature.startswith("path:sentence:"):
            return False
        return True

    @staticmethod
    def _feature_weight(feature: str, *, query_terms: set[str], current: bool) -> float:
        if feature.startswith("path:"):
            # Path coverage is valuable, but it is a structural quality
            # signal.  Lexical answer anchors receive higher weight so an
            # extra bridge entity cannot make the quality floor unattainable.
            return 0.20
        if feature.startswith("doc:"):
            return 0.55 if current else 0.35
        if feature.startswith("sentence:"):
            return 0.18
        if feature in query_terms:
            if _NUMBER_OR_DATE.fullmatch(feature) or feature in _NEGATIONS:
                return 2.4
            return 1.35
        return 0.40

    def _filter_hits(
        self,
        request: WorkloadRequest,
        hits: list[EvidenceHit],
        current_ids: set[str],
    ) -> tuple[list[EvidenceHit], dict[str, Any]]:
        query_terms = _feature_tokens(request.question) - _QUESTION_WORDS
        accepted: list[EvidenceHit] = []
        historical: list[EvidenceHit] = []
        for hit in hits:
            if hit.node_id in current_ids:
                accepted.append(hit)
                continue
            lexical = len(query_terms & _feature_tokens(hit.text)) / max(1, len(query_terms))
            # Historical evidence needs either a graph bridge or a meaningful
            # lexical link.  This avoids importing unrelated session material.
            if hit.hops <= self.max_hops and (hit.path or lexical >= 0.10):
                historical.append(hit)
        historical.sort(key=lambda h: (-h.score, h.hops, h.node_id))
        max_history = max(1, int(self.top_k_candidates * self.max_historical_fraction))
        accepted.extend(historical[:max_history])
        return accepted, {
            "historical_gate_rejected_count": max(0, len(historical) - max_history),
            "historical_gate_accepted_count": len(historical[:max_history]),
            "quality_floor": self.quality_floor,
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
        # Reuse lowers the budget only after the quality core is protected.
        reuse_signal = min(1.0, len(historical_hits) / 8.0)
        query_signal = min(1.0, len(_feature_tokens(query)) / 24.0)
        ratio = self.max_retain_ratio - 0.18 * reuse_signal + 0.06 * query_signal
        if original_tokens < 128:
            ratio = max(ratio, 0.55)
        return max(self.min_retain_ratio, min(self.max_retain_ratio, ratio))

    def _pack_qc(
        self,
        request: WorkloadRequest,
        hits: list[EvidenceHit],
        current_ids: set[str],
        *,
        tokenizer: Tokenizer,
        budget: int,
    ) -> tuple[str, list[EvidenceHit], dict[str, Any]]:
        prefix = "Answer the question using only the retrieved context.\nContext:\n"
        suffix = f"\nQuestion: {request.question}\nAnswer:"
        fixed = tokenizer.count_tokens(prefix + suffix)
        available = max(0, budget - fixed)
        if not hits or available <= 0:
            return prefix + suffix, [], {
                "quality_proxy_full": 0.0,
                "quality_proxy_selected": 0.0,
                "quality_proxy_ratio": 0.0,
                "quality_core_count": 0,
                "quality_core_complete": False,
            }

        query_features = self._query_features(request.question, hits)
        feature_sets = [self._candidate_features(hit, query_features) for hit in hits]
        candidate_costs = [tokenizer.count_tokens((f"[{h.title}] " if h.title else "") + h.text + "\n") for h in hits]
        current_flags = [h.node_id in current_ids for h in hits]
        weights = {
            feature: self._feature_weight(feature, query_terms=query_features, current=True)
            for feature in set().union(*feature_sets)
        }
        full_features = {feature for features in feature_sets for feature in features
                         if self._is_quality_feature(feature)}
        full_quality = sum(weights.get(feature, 0.4) for feature in full_features)
        target_quality = full_quality * self.quality_floor

        selected: list[int] = []
        selected_features: set[str] = set()
        used = 0
        core_complete = True

        def score(i: int, *, core: bool = False) -> tuple[float, float, float, int]:
            new = feature_sets[i] - selected_features
            gain = sum(weights.get(feature, 0.4) for feature in new)
            current_bonus = 0.22 if current_flags[i] else 0.0
            bridge_bonus = 0.18 * max(0, len(hits[i].path) - 1)
            cost = max(1, candidate_costs[i])
            return (gain / cost + current_bonus / cost + bridge_bonus / cost,
                    gain if core else 0.0, -float(cost), -i)

        # Core pass: cover query-bearing terms and bridge paths with the best
        # compact sentence.  Numeric/date/negation features are mandatory when
        # present because dropping them has a high distortion cost.
        mandatory = {
            feature for feature in full_features
            if feature in query_features and (
                _NUMBER_OR_DATE.fullmatch(feature) or feature in _NEGATIONS
            )
        }
        for feature in sorted(mandatory):
            options = [i for i, fs in enumerate(feature_sets)
                       if feature in fs and i not in selected]
            if not options:
                core_complete = False
                continue
            feasible = [i for i in options if used + candidate_costs[i] <= available]
            if not feasible:
                core_complete = False
                continue
            best = max(feasible, key=lambda i: score(i, core=True))
            selected.append(best)
            used += candidate_costs[best]
            selected_features.update(feature_sets[best])

        # Ensure at least one current sentence is retained whenever the budget
        # allows it.  Current raw evidence is the strongest provenance anchor.
        if not any(current_flags[i] for i in selected):
            current_options = [i for i, flag in enumerate(current_flags)
                               if flag and i not in selected and used + candidate_costs[i] <= available]
            if current_options:
                best = max(current_options, key=lambda i: score(i, core=True))
                selected.append(best)
                used += candidate_costs[best]
                selected_features.update(feature_sets[best])
            else:
                core_complete = False

        # Rate-distortion pass.  The loop is deterministic and uses no model
        # call, so compression overhead stays proportional to candidate count.
        remaining = set(range(len(hits))) - set(selected)
        while remaining and used < available:
            feasible = [i for i in remaining if used + candidate_costs[i] <= available]
            if not feasible:
                break
            best = max(feasible, key=lambda i: score(i))
            gain = sum(
                weights.get(feature, 0.4)
                for feature in feature_sets[best] - selected_features
                if self._is_quality_feature(feature)
            )
            selected_quality_now = sum(
                weights.get(feature, 0.4)
                for feature in selected_features
                if self._is_quality_feature(feature)
            )
            if selected_quality_now >= target_quality and gain <= 0:
                break
            selected.append(best)
            remaining.remove(best)
            used += candidate_costs[best]
            selected_features.update(feature_sets[best])
            quality_now = sum(
                weights.get(feature, 0.4)
                for feature in selected_features
                if self._is_quality_feature(feature)
            )
            if quality_now >= target_quality:
                # One additional high-rate symbol can preserve a second source
                # without materially changing latency; the next iteration will
                # stop when its marginal gain is zero.
                continue

        selected = sorted(selected, key=lambda i: (
            hits[i].source_document_id or "",
            hits[i].source_sentence_index if hits[i].source_sentence_index is not None else 0,
            hits[i].node_id,
        ))
        selected_hits = [hits[i] for i in selected]
        context = "\n".join((f"[{h.title}] " if h.title else "") + h.text for h in selected_hits)
        prompt = prefix + context + suffix

        # If an individual source is longer than the available budget, retain a
        # tokenizer-safe prefix while preserving the question and delimiter.
        if tokenizer.count_tokens(prompt) > budget and selected_hits:
            first = selected_hits[0]
            evidence_budget = max(0, available - tokenizer.count_tokens(f"[{first.title}] " if first.title else ""))
            clipped = tokenizer.truncate_to_tokens(first.text, evidence_budget)
            selected_hits = [replace(first, text=clipped)] if clipped else []
            context = "\n".join((f"[{h.title}] " if h.title else "") + h.text for h in selected_hits)
            prompt = prefix + context + suffix

        selected_quality = sum(
            weights.get(feature, 0.4)
            for feature in selected_features
            if self._is_quality_feature(feature)
        )
        selected_quality_features = {
            feature for feature in selected_features if self._is_quality_feature(feature)
        }
        audit = {
            "quality_proxy_full": round(full_quality, 6),
            "quality_proxy_selected": round(selected_quality, 6),
            "quality_proxy_ratio": round(selected_quality / max(full_quality, 1e-9), 6),
            "quality_target_ratio": self.quality_floor,
            "quality_core_count": len(selected),
            "quality_core_complete": core_complete,
            "quality_feature_count_full": len(full_features),
            "quality_feature_count_selected": len(selected_quality_features),
            "path_count_full": sum(bool(hit.path) for hit in hits),
            "path_count_selected": sum(bool(hit.path) for hit in selected_hits),
        }
        return prompt, selected_hits, audit

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
            _, current_sentence_ids = ingest_request_context(
                request, graph=self.graph, turn_id=request.request_id, status="uncertain"
            )
            current_ids = set(current_sentence_ids)
            # Use all relevant graph hits for the quality reference.  The graph
            # itself remains bounded by top_k_candidates for routine requests.
            hits = self.graph.search(
                request.question,
                query_entities=_extract_entities(request.question, ""),
                top_k=max(self.top_k_candidates, len(current_ids)),
                max_hops=self.max_hops,
            )
            hits, filter_metadata = self._filter_hits(request, hits, current_ids)
            historical_hits = [hit for hit in hits if hit.node_id not in current_ids]
            ratio = self._choose_ratio(
                target_budget_ratio,
                original_tokens=original_tokens,
                historical_hits=historical_hits,
                query=request.question,
            )
            budget = max(1, int(original_tokens * ratio))
            prompt, selected_hits, audit = self._pack_qc(
                request, hits, current_ids, tokenizer=tokenizer, budget=budget
            )

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
            "packed_evidence_tokens": tokenizer.count_tokens("\n".join(hit.text for hit in selected_hits)),
            "selected_sentence_ids": [hit.node_id for hit in selected_hits],
            "selected_source_document_ids": sorted({hit.source_document_id for hit in selected_hits if hit.source_document_id}),
            "selected_paths": [list(hit.path) for hit in selected_hits],
            "compression_objective": "minimize token cost subject to observable graph quality floor",
        }
        metadata.update(filter_metadata)
        metadata.update(audit)
        return CompressionRecord(
            method=self.name,
            original_prompt_tokens=original_tokens,
            compressed_prompt_tokens=tokenizer.count_tokens(prompt),
            compressed_prompt=prompt,
            latency_ms=measured[0],
            target_budget_ratio=target_budget_ratio,
            metadata=metadata,
        )


# Short alias for callers that want the algorithm name without the full class.
QCGRDCCompressor = QualityConstrainedGraphRateDistortionCompressor
