from ladc.quality import evaluate_real_request
from ladc.workloads import WorkloadRequest


def _request() -> WorkloadRequest:
    context = "[Film] Alice starred in Blue Horizon.\n[Director] Blue Horizon was directed by Bob Stone.\n[Noise] Unrelated text."
    return WorkloadRequest(
        request_id="q1",
        prompt=f"Context:\n{context}\nQuestion: Who directed Blue Horizon?\nAnswer:",
        question="Who directed Blue Horizon?",
        context=context,
        references=("Bob Stone",),
        supporting_facts=(("Director", 0),),
        metadata={"dataset": "toy"},
    )


def test_evidence_metrics_measure_supporting_sentence_not_only_title():
    metrics = evaluate_real_request(
        _request(),
        "Context:\n[Director] Blue Horizon was directed by Bob Stone.\nQuestion: Who directed Blue Horizon?\nAnswer:",
        "Bob Stone",
    )

    assert metrics["supporting_fact_precision"] == 1.0
    assert metrics["supporting_fact_recall"] == 1.0
    assert metrics["evidence_f1"] == 1.0
    assert metrics["answer_context_support"] == 1.0
    assert "citation_f1" not in metrics


def test_evidence_metrics_penalize_retaining_only_irrelevant_evidence():
    metrics = evaluate_real_request(
        _request(),
        "Context:\n[Noise] Unrelated text.\nQuestion: Who directed Blue Horizon?\nAnswer:",
        "Bob Stone",
    )

    assert metrics["supporting_fact_recall"] == 0.0
    assert metrics["supporting_fact_precision"] == 0.0
