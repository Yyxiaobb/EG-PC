"""Quality metrics shared by synthetic and real-model evaluations."""

from __future__ import annotations

import re
import string
from collections import Counter

from ladc.evidence_builder import _split_blocks, _split_sentences
from ladc.workloads import WorkloadRequest


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(reference) for reference in references))


def token_f1(prediction: str, references: list[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0
    best = 0.0
    for reference in references:
        reference_tokens = normalize_answer(reference).split()
        if not reference_tokens:
            continue
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def evaluate_generation(prediction: str, references: list[str]) -> dict[str, float]:
    """Return standard answer metrics for one generated response."""

    em = exact_match(prediction, references)
    f1 = token_f1(prediction, references)
    return {"exact_match": em, "token_f1": f1, "primary_quality": f1}


def _context_from_prompt(prompt: str) -> str:
    match = re.search(r"Context:\s*(.*?)(?:\nQuestion:|$)", prompt, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else prompt


def _evidence_items(context: str) -> list[tuple[str, int, str]]:
    items: list[tuple[str, int, str]] = []
    for title, block in _split_blocks(context):
        for index, sentence in enumerate(_split_sentences(block)):
            items.append((title, index, sentence))
    return items


def _selected_evidence_items(prompt: str) -> list[tuple[str, int, str]]:
    return _evidence_items(_context_from_prompt(prompt))


def _normalized_contains(container: str, content: str) -> bool:
    normalized_container = normalize_answer(container)
    normalized_content = normalize_answer(content)
    return bool(normalized_content) and normalized_content in normalized_container


def _supporting_fact_metrics(
    request: WorkloadRequest,
    compressed_prompt: str,
) -> dict[str, float]:
    """Measure offline evidence retention against supporting annotations.

    The metric is computed after compression and is never passed into the
    compressor.  If a dataset annotation points to a paragraph rather than a
    sentence, title-level matching is used as a conservative fallback.
    """

    selected = _selected_evidence_items(compressed_prompt)
    if not request.supporting_facts:
        return {
            "supporting_fact_precision": 0.0,
            "supporting_fact_recall": 0.0,
            "evidence_f1": 0.0,
            "gold_supporting_fact_count": 0.0,
            "selected_evidence_count": float(len(selected)),
        }
    source = _evidence_items(request.context)
    selected_matches: set[int] = set()
    gold_matches: list[set[int]] = []
    for title, fact_index in request.supporting_facts:
        candidates = {
            source_index
            for source_index, (source_title, sentence_index, _) in enumerate(source)
            if normalize_answer(source_title) == normalize_answer(title)
            and sentence_index == fact_index
        }
        if not candidates:
            candidates = {
                source_index
                for source_index, (source_title, _, _) in enumerate(source)
                if normalize_answer(source_title) == normalize_answer(title)
            }
        gold_matches.append(candidates)
    for selected_title, _, selected_text in selected:
        for source_index, (source_title, _, source_text) in enumerate(source):
            if normalize_answer(selected_title) != normalize_answer(source_title):
                continue
            if _normalized_contains(selected_text, source_text) or _normalized_contains(source_text, selected_text):
                selected_matches.add(source_index)
    gold_indices = set().union(*gold_matches) if gold_matches else set()
    retained = len(selected_matches & gold_indices)
    precision = retained / max(1, len(selected_matches))
    recall = retained / max(1, len(gold_indices))
    evidence_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "supporting_fact_precision": precision,
        "supporting_fact_recall": recall,
        "evidence_f1": evidence_f1,
        "gold_supporting_fact_count": float(len(gold_indices)),
        "selected_evidence_count": float(len(selected_matches)),
    }


def evaluate_context_preservation(prompt: str, expected_answer: str) -> dict[str, float]:
    """Quality proxy for synthetic tests before a real generator is connected."""

    present = float(normalize_answer(expected_answer) in normalize_answer(prompt))
    return {
        "answer_presence": present,
        "context_quality": present,
        "primary_quality": present,
    }


def evaluate_real_request(
    request: WorkloadRequest,
    compressed_prompt: str,
    output_text: str,
) -> dict[str, float]:
    """Evaluate answer quality and source-backed evidence preservation."""

    answer_metrics = evaluate_generation(output_text, list(request.references))
    supporting_metrics = _supporting_fact_metrics(request, compressed_prompt)
    answer_context_support = float(
        bool(output_text.strip())
        and any(
            token in normalize_answer(compressed_prompt)
            for token in normalize_answer(output_text).split()
            if len(token) > 2
        )
    )
    # This is intentionally conservative: exact-match answers are treated as
    # faithful, while non-exact generations require later entailment judging.
    faithfulness = answer_metrics["exact_match"]
    return {
        **answer_metrics,
        "answer_context_support": answer_context_support,
        **supporting_metrics,
    }
