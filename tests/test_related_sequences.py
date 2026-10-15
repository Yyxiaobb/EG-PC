from ladc.workloads import WorkloadRequest
from scripts.run_real_7b_related_sequences import build_related_sequences


def _request(request_id: str, titles: tuple[str, ...]) -> WorkloadRequest:
    return WorkloadRequest(
        request_id=request_id,
        prompt="prompt",
        question="question",
        context="context",
        references=("answer",),
        metadata={"context_titles": titles, "dataset": "toy"},
    )


def test_related_sequence_requires_shared_context_title():
    requests = {
        "toy": [
            _request("a", ("A", "B")),
            _request("b", ("B", "C")),
            _request("c", ("C", "D")),
            _request("d", ("X",)),
        ]
    }

    sequences = build_related_sequences(requests, sequence_length=3, max_sequences_per_dataset=1)

    assert len(sequences) == 1
    assert [request.request_id for request in sequences[0][1]] == ["a", "b", "c"]
