"""Query-aware extractive compression baseline.

This is a dependency-free BM25-style sentence ranker with MMR diversity.  It
is a stronger and more reproducible baseline than prefix/fixed-ratio cutting,
while keeping compression overhead measurable on the same machine.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha1
from math import floor, log
import re
from typing import Any

from ladc.evidence_builder import _split_blocks, _split_sentences
from ladc.evidence_graph import _tokens
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


class QueryAwareMMRCompressor:
    """Extractive query-aware compression without cross-request memory."""

    name = "query_aware_bm25_mmr"

    def __init__(
        self,
        *,
        retain_ratio: float = 0.5,
        mmr_lambda: float = 0.75,
        name: str | None = None,
    ) -> None:
        _validate_ratio(retain_ratio)
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        self.retain_ratio = retain_ratio
        self.mmr_lambda = mmr_lambda
        if name is not None:
            self.name = name

    def compress(
        self,
        prompt: str,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
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
            ratio = self.retain_ratio if target_budget_ratio is None else target_budget_ratio
            _validate_ratio(ratio)
            budget = max(1, floor(original_tokens * ratio))
            prefix = "Answer the question using only the retrieved context.\nContext:\n"
            suffix = f"\nQuestion: {request.question}\nAnswer:"
            available = max(1, budget - tokenizer.count_tokens(prefix + suffix))
            candidates = self._candidates(request)
            selected = self._select(candidates, request.question, tokenizer, available)
            context = "\n".join(
                f"[{title}] {text}" if title else text
                for title, _, text, _ in selected
            )
            compressed_prompt = prefix + context + suffix
            if tokenizer.count_tokens(compressed_prompt) > budget:
                compressed_prompt = tokenizer.truncate_to_tokens(compressed_prompt, budget)
        compressed_tokens = tokenizer.count_tokens(compressed_prompt)
        return CompressionRecord(
            method=self.name,
            original_prompt_tokens=original_tokens,
            compressed_prompt_tokens=compressed_tokens,
            compressed_prompt=compressed_prompt,
            latency_ms=measured[0],
            target_budget_ratio=target_budget_ratio,
            metadata={
                "candidate_evidence_count": len(candidates),
                "selected_evidence_count": len(selected),
                "budget_tokens": budget,
                "mmr_lambda": self.mmr_lambda,
                "dynamic_budget_ratio": ratio,
                "historical_candidate_count": 0,
                "reused_evidence_count": 0,
            },
        )

    def _candidates(self, request: WorkloadRequest) -> list[tuple[str, int, str, set[str]]]:
        candidates: list[tuple[str, int, str, set[str]]] = []
        for title, block in _split_blocks(request.context, request.metadata.get("context_titles", ())):
            for sentence_index, sentence in enumerate(_split_sentences(block)):
                candidates.append((title, sentence_index, sentence, _tokens(f"{title} {sentence}")))
        return candidates

    def _bm25_scores(
        self,
        candidates: list[tuple[str, int, str, set[str]]],
        query: str,
    ) -> list[float]:
        query_terms = _tokens(query)
        if not candidates or not query_terms:
            return [0.0] * len(candidates)
        document_frequency = Counter(
            term
            for _, _, _, terms in candidates
            for term in terms
        )
        average_length = sum(len(terms) for _, _, _, terms in candidates) / len(candidates)
        scores: list[float] = []
        k1 = 1.2
        b = 0.75
        for _, _, _, terms in candidates:
            length = len(terms)
            score = 0.0
            for term in query_terms:
                if term not in terms:
                    continue
                df = document_frequency[term]
                idf = log(1.0 + (len(candidates) - df + 0.5) / (df + 0.5))
                tf = 1.0
                normalization = 1.0 - b + b * length / max(1.0, average_length)
                score += idf * (tf * (k1 + 1.0) / (tf + k1 * normalization))
            scores.append(score)
        maximum = max(scores, default=0.0)
        return [score / maximum if maximum else 0.0 for score in scores]

    def _select(
        self,
        candidates: list[tuple[str, int, str, set[str]]],
        query: str,
        tokenizer: Tokenizer,
        available_tokens: int,
    ) -> list[tuple[str, int, str, set[str]]]:
        scores = self._bm25_scores(candidates, query)
        selected_indices: list[int] = []
        used = 0
        remaining = set(range(len(candidates)))
        while remaining:
            best_index: int | None = None
            best_score = float("-inf")
            for index in remaining:
                title, _, text, terms = candidates[index]
                line = f"[{title}] {text}" if title else text
                cost = tokenizer.count_tokens(line + "\n")
                if selected_indices and used + cost > available_tokens:
                    continue
                redundancy = 0.0
                if selected_indices:
                    redundancy = max(
                        self._jaccard(terms, candidates[other][3])
                        for other in selected_indices
                    )
                mmr_score = self.mmr_lambda * scores[index] - (1.0 - self.mmr_lambda) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index
            if best_index is None:
                break
            selected_indices.append(best_index)
            title, _, text, _ = candidates[best_index]
            used += tokenizer.count_tokens((f"[{title}] {text}" if title else text) + "\n")
            remaining.remove(best_index)
            if used >= available_tokens:
                break
        selected_indices.sort(key=lambda index: (candidates[index][0], candidates[index][1]))
        return [candidates[index] for index in selected_indices]

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 0.0
