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
from functools import lru_cache
import re
from typing import Any, Iterable

from ladc.compressors.evidence_graph import EvidenceGraphCompressor, _validate_ratio
from ladc.compressors.query_aware import QueryAwareMMRCompressor
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

# Lightweight, deterministic relation vocabulary for relation-aware evidence
# selection. This is deliberately a prototype rather than a general-purpose
# open IE system; source sentences remain the authority and retain provenance.
_RELATION_PATTERNS = {
    "directed": (r"\bdirected\s+by\b", r"\bdirect(?:s|ed)?\b", r"\bdirector\b"),
    "written": (r"\bwritten\s+by\b", r"\bwrote\b", r"\bwriter\b", r"\bauthor\b"),
    "born_in": (r"\bborn\s+in\b",),
    "located_in": (r"\blocated\s+in\b", r"\bbased\s+in\b", r"\bsituated\s+in\b"),
    "capital_of": (r"\bcapital\s+of\b",),
    "released_in": (r"\breleased\s+in\b", r"\bcame\s+out\s+in\b"),
    "starred_in": (r"\bstarr?ed\s+in\b", r"\bappeared\s+in\b"),
    "produced": (r"\bproduced\s+by\b", r"\bproducer\b"),
    "founded": (r"\bfounded\s+by\b", r"\bfounded\b"),
    "member_of": (r"\bmember\s+of\b", r"\bbelongs\s+to\b"),
    "spouse_of": (r"\bmarried\s+to\b", r"\bspouse\b", r"\bwife\b", r"\bhusband\b"),
    "parent_of": (r"\bson\s+of\b", r"\bdaughter\s+of\b", r"\bfather\s+of\b", r"\bmother\s+of\b"),
    "died_in": (r"\bdied\s+in\b", r"\bkilled\s+in\b"),
    "educated_at": (r"\beducated\s+at\b", r"\bgraduated\s+from\b"),
    "played_for": (r"\bplayed\s+for\b", r"\bplays\s+for\b"),
    "competed_in": (r"\bcompeted\s+in\b", r"\bparticipated\s+in\b"),
}
_RELATION_REGEX = {label: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
                   for label, patterns in _RELATION_PATTERNS.items()}


def _relation_labels(text: str) -> set[str]:
    return {label for label, patterns in _RELATION_REGEX.items()
            if any(pattern.search(text) for pattern in patterns)}

_TRIPLE_PATTERNS = {
    "directed": re.compile(r"^(?P<subj>.+?)\s+(?:was\s+)?directed\s+by\s+(?P<obj>[^,.!?;]+)", re.I),
    "directed_active": re.compile(r"^(?P<subj>.+?)\s+directed\s+(?!by\b)(?P<obj>[^,.!?;]+)", re.I),
    "written": re.compile(r"^(?P<subj>.+?)\s+(?:was\s+)?written\s+by\s+(?P<obj>[^,.!?;]+)", re.I),
    "born_in": re.compile(r"^(?P<subj>.+?)\s+was\s+born\s+in\s+(?P<obj>[^,.!?;]+)", re.I),
    "located_in": re.compile(r"^(?P<subj>.+?)\s+(?:is|was)\s+(?:located|based|situated)\s+in\s+(?P<obj>[^,.!?;]+)", re.I),
    "capital_of": re.compile(r"^(?P<subj>.+?)\s+is\s+the\s+capital\s+of\s+(?P<obj>[^,.!?;]+)", re.I),
    "released_in": re.compile(r"^(?P<subj>.+?)\s+was\s+released\s+in\s+(?P<obj>\d{3,4})", re.I),
    "starred_in": re.compile(r"^(?P<subj>.+?)\s+starred\s+in\s+(?P<obj>[^,.!?;]+)", re.I),
    "produced": re.compile(r"^(?P<subj>.+?)\s+was\s+produced\s+by\s+(?P<obj>[^,.!?;]+)", re.I),
    "founded": re.compile(r"^(?P<subj>.+?)\s+was\s+founded\s+by\s+(?P<obj>[^,.!?;]+)", re.I),
    "member_of": re.compile(r"^(?P<subj>.+?)\s+is\s+a?\s*member\s+of\s+(?P<obj>[^,.!?;]+)", re.I),
    "educated_at": re.compile(r"^(?P<subj>.+?)\s+was\s+educated\s+at\s+(?P<obj>[^,.!?;]+)", re.I),
    "played_for": re.compile(r"^(?P<subj>.+?)\s+played\s+for\s+(?P<obj>[^,.!?;]+)", re.I),
}

@lru_cache(maxsize=32768)
def _extract_relation_triples(title: str, text: str) -> tuple[tuple[str, str, str], ...]:
    """Extract conservative, source-backed triples from simple English clauses."""
    triples = []
    clean = " ".join(text.strip().split()).strip(" .!?;")
    for relation, pattern in _TRIPLE_PATTERNS.items():
        match = pattern.search(clean)
        if not match:
            continue
        normalized_relation = "directed" if relation == "directed_active" else relation
        subject = (title.strip() or match.group("subj")).strip(" []")
        obj = match.group("obj").strip(" []")
        if subject and obj:
            triples.append((subject, normalized_relation, obj))
    return tuple(triples)

def _entity_keys(value: str) -> set[str]:
    tokens = [t for t in re.findall(r"[a-z0-9]+", value.casefold()) if len(t) > 2]
    return {" ".join(tokens), *tokens} if tokens else set()


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
        adaptive_quality_budget: bool = True,
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
        self.adaptive_quality_budget = adaptive_quality_budget
        self._ranker = QueryAwareMMRCompressor(mmr_lambda=0.80)

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
        if hit.hops >= 2 and any(node_id.startswith("entity:") for node_id in hit.path):
            features.add(f"hop:{min(hit.hops, 4)}")
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
        if feature.startswith("hop:"):
            return 1.0
        if feature.startswith("doc:"):
            return 0.55 if current else 0.35
        if feature.startswith("sentence:"):
            return 0.18
        if feature in query_terms:
            if _NUMBER_OR_DATE.fullmatch(feature) or feature in _NEGATIONS:
                return 2.4
            return 1.35
        return 0.40

    def _relation_chain_bundles(self, hits: list[EvidenceHit]) -> list[tuple[int, int, str]]:
        """Optional pairs of sentences that jointly encode a relation path."""
        return []

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
                # Question/answer nodes are contextual bookkeeping, not source
                # evidence.  A path that crosses one is flattened for current
                # material so it cannot create a fake bridge feature.
                if any(part.startswith(("question:", "answer:")) for part in hit.path):
                    accepted.append(replace(hit, path=(hit.node_id,), hops=0))
                else:
                    accepted.append(hit)
                continue
            lexical = len(query_terms & _feature_tokens(hit.text)) / max(1, len(query_terms))
            # Historical evidence needs either a graph bridge or a meaningful
            # lexical link.  This avoids importing unrelated session material.
            if any(part.startswith(("question:", "answer:")) for part in hit.path):
                continue
            has_bridge = any(part.startswith("entity:") for part in hit.path[::2])
            if hit.hops <= self.max_hops and (has_bridge or lexical >= 0.10):
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
        rank_candidates = [
            (hit.title, hit.source_sentence_index or 0, hit.text, _feature_tokens(hit.title + " " + hit.text))
            for hit in hits
        ]
        bm25_scores = self._ranker._bm25_scores(rank_candidates, request.question)
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
            # The graph search score is the answer-bearing relevance signal;
            # include it in the rate term so generic feature overlap cannot
            # outrank a high-scoring supporting sentence.
            relevance_gain = 1.50 * max(0.0, hits[i].score) + 1.50 * bm25_scores[i]
            current_bonus = 0.22 if current_flags[i] else 0.0
            bridge_bonus = 0.18 * max(0, len(hits[i].path) - 1)
            cost = max(1, candidate_costs[i])
            return ((gain + relevance_gain) / cost + current_bonus / cost + bridge_bonus / cost,
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
        # A multi-hop candidate contributes a structural depth feature.  Keep
        # one compact high-confidence sentence for every observed depth so the
        # selector cannot satisfy lexical coverage with only direct matches.
        mandatory.update(feature for feature in full_features if feature.startswith("hop:"))
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

        # Relation-aware variants may reserve a complete two-edge path as a
        # bundle. A path is only counted when both source sentences fit.
        selected_chain_ids: list[str] = []
        bundles = self._relation_chain_bundles(hits)
        if bundles:
            bundle_options = []
            for left, right, chain_id in bundles:
                missing = [i for i in (left, right) if i not in selected]
                cost = sum(candidate_costs[i] for i in missing)
                if cost and used + cost <= available:
                    value = sum(score(i)[0] for i in missing) / max(1, cost)
                    bundle_options.append((value, -cost, chain_id, missing))
                elif not missing:
                    selected_chain_ids.append(chain_id)
            if bundle_options:
                _, _, chain_id, bundle = max(bundle_options)
                for i in bundle:
                    selected.append(i)
                    used += candidate_costs[i]
                    selected_features.update(feature_sets[i])
                selected_chain_ids.append(chain_id)

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
            "selected_relation_chain_ids": selected_chain_ids,
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
            requested_budget = budget
            budget_expanded = False
            # A hard budget can leave no room for the question's fixed prompt
            # wrapper or a protected answer-bearing sentence.  Since this is a
            # quality-constrained method, expand only that request's budget
            # until the observable floor is met (or the original prompt is the
            # only feasible representation).  The expansion is audited rather
            # than silently violating the caller's requested ratio.
            if (
                self.adaptive_quality_budget
                and hits
                and (
                    not audit.get("quality_core_complete", False)
                    or float(audit.get("quality_proxy_ratio", 0.0)) < self.quality_floor
                )
                and budget < original_tokens
            ):
                prefix = "Answer the question using only the retrieved context.\nContext:\n"
                suffix = f"\nQuestion: {request.question}\nAnswer:"
                fixed = tokenizer.count_tokens(prefix + suffix)
                smallest = min(
                    tokenizer.count_tokens((f"[{hit.title}] " if hit.title else "") + hit.text + "\n")
                    for hit in hits
                )
                expanded_budget = min(
                    original_tokens,
                    max(budget + 1, int(original_tokens * max(ratio, self.quality_floor)), fixed + smallest),
                )
                if expanded_budget > budget:
                    prompt, selected_hits, audit = self._pack_qc(
                        request, hits, current_ids, tokenizer=tokenizer, budget=expanded_budget
                    )
                    budget = expanded_budget
                    budget_expanded = True

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
            "requested_budget_tokens": requested_budget,
            "budget_expanded_for_quality": budget_expanded,
            "packed_evidence_tokens": tokenizer.count_tokens("\n".join(hit.text for hit in selected_hits)),
            "selected_sentence_ids": [hit.node_id for hit in selected_hits],
            "selected_source_document_ids": sorted({hit.source_document_id for hit in selected_hits if hit.source_document_id}),
            "selected_paths": [list(hit.path) for hit in selected_hits],
            "compression_objective": "minimize token cost subject to observable graph quality floor",
        }
        metadata.update(filter_metadata)
        metadata.update(audit)
        if isinstance(self, TripleAwareQCGRDCCompressor):
            metadata.update(relation_triple_audit(hits))
        if isinstance(self, ChainAwareQCGRDCCompressor):
            metadata["candidate_relation_chain_count"] = len(self._relation_chain_bundles(hits))
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


class RelationAwareQCGRDCCompressor(QualityConstrainedGraphRateDistortionCompressor):
    """QC-GRDC variant that favors sentences matching query relation types.

    Relations are recognized with deterministic cached regex patterns. The
    extractor contributes a relation feature to the existing marginal-gain
    objective, so matching predicates and multi-sentence predicate chains can
    be retained without changing the original v4 compressor.
    """

    name = "evidence_graph_qc_grdc_relation_v1"

    @staticmethod
    def _query_features(question: str, hits: Iterable[EvidenceHit]) -> set[str]:
        features = QualityConstrainedGraphRateDistortionCompressor._query_features(question, hits)
        features.update(f"relation:{label}" for label in _relation_labels(question))
        return features

    @staticmethod
    def _candidate_features(hit: EvidenceHit, query_features: set[str]) -> set[str]:
        features = QualityConstrainedGraphRateDistortionCompressor._candidate_features(hit, query_features)
        features.update(f"relation:{label}" for label in _relation_labels(hit.text))
        return features

    @staticmethod
    def _is_quality_feature(feature: str) -> bool:
        return QualityConstrainedGraphRateDistortionCompressor._is_quality_feature(feature)

    @staticmethod
    def _feature_weight(feature: str, *, query_terms: set[str], current: bool) -> float:
        if feature.startswith("relation:"):
            return 2.8 if feature in query_terms else 0.4
        return QualityConstrainedGraphRateDistortionCompressor._feature_weight(
            feature, query_terms=query_terms, current=current
        )


class TripleAwareQCGRDCCompressor(RelationAwareQCGRDCCompressor):
    """Add conservative source-backed subject-relation-object features."""

    name = "evidence_graph_qc_grdc_triple_v1"

    @staticmethod
    def _candidate_features(hit: EvidenceHit, query_features: set[str]) -> set[str]:
        features = RelationAwareQCGRDCCompressor._candidate_features(hit, query_features)
        for subject, relation, obj in _extract_relation_triples(hit.title, hit.text):
            features.add(f"triple:{_normalize(subject)}|{relation}|{_normalize(obj)}")
            features.update(f"triple_entity:{entity}" for entity in _entity_keys(subject))
            features.update(f"triple_entity:{entity}" for entity in _entity_keys(obj))
        return features

    @staticmethod
    def _feature_weight(feature: str, *, query_terms: set[str], current: bool) -> float:
        if feature.startswith("triple:"):
            return 0.25
        if feature.startswith("triple_entity:"):
            return 0.5
        return RelationAwareQCGRDCCompressor._feature_weight(
            feature, query_terms=query_terms, current=current
        )


class ChainAwareQCGRDCCompressor(TripleAwareQCGRDCCompressor):
    """Prefer complete two-edge relation paths that share an entity."""

    name = "evidence_graph_qc_grdc_chain_v1"

    def _relation_chain_bundles(self, hits: list[EvidenceHit]) -> list[tuple[int, int, str]]:
        triples = []
        for index, hit in enumerate(hits):
            for subject, relation, obj in _extract_relation_triples(hit.title, hit.text):
                triples.append((index, subject, relation, obj))
        bundles: dict[tuple[int, int], str] = {}
        for pos, (left, subject_a, relation_a, object_a) in enumerate(triples):
            keys_a = _entity_keys(subject_a) | _entity_keys(object_a)
            for right, subject_b, relation_b, object_b in triples[pos + 1:]:
                if left == right:
                    continue
                keys_b = _entity_keys(subject_b) | _entity_keys(object_b)
                shared = keys_a & keys_b
                # Ignore single-token matches: common surnames/words create
                # spurious graph links in this conservative prototype.
                shared = {key for key in shared if " " in key}
                if not shared:
                    continue
                pair = tuple(sorted((left, right)))
                chain_id = f"{pair[0]}>{pair[1]}:{relation_a}>{relation_b}:{sorted(shared)[0]}"
                bundles[pair] = chain_id
        return [(left, right, chain_id)
                for (left, right), chain_id in sorted(bundles.items())]


def relation_triple_audit(hits: list[EvidenceHit]) -> dict[str, Any]:
    triples = []
    for hit in hits:
        for subject, relation, obj in _extract_relation_triples(hit.title, hit.text):
            triples.append({
                "subject": subject, "relation": relation, "object": obj,
                "sentence_id": hit.node_id,
            })
    return {"extracted_relation_triple_count": len(triples),
            "relation_triples": triples}
