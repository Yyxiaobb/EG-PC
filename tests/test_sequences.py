from ladc.sequences import build_related_sequence_specs, build_unrelated_sequence_specs, relation_features
from ladc.workloads import WorkloadRequest


def _request(request_id: str, titles: tuple[str, ...], question: str = "Question") -> WorkloadRequest:
    return WorkloadRequest(
        request_id=request_id,
        prompt="prompt",
        question=question,
        context="context",
        references=("answer",),
        metadata={"context_titles": titles},
    )


def test_related_sequences_require_adjacent_visible_links():
    requests = {
        "toy": [
            _request("a", ("A",)),
            _request("b", ("A", "B")),
            _request("c", ("B", "C")),
        ]
    }

    specs = build_related_sequence_specs(requests, sequence_length=3, max_sequences_per_dataset=1)

    assert len(specs) == 1
    assert [request.request_id for request in specs[0].requests] == ["a", "b", "c"]
    assert specs[0].adjacent_relations[0]["shared_titles"] == ["a"]
    assert specs[0].adjacent_relations[1]["shared_titles"] == ["b"]


def test_unrelated_sequences_have_no_title_or_query_entity_links():
    requests = {
        "toy": [
            _request("a", ("A",)),
            _request("b", ("B",)),
            _request("c", ("C",)),
        ]
    }

    specs = build_unrelated_sequence_specs(requests, sequence_length=3, max_sequences_per_dataset=1)

    assert len(specs) == 1
    assert all(not values for relation in specs[0].adjacent_relations for values in relation.values())


def test_relation_features_do_not_use_gold_annotations():
    left = _request("a", ("A",))
    right = _request("b", ("B",))
    assert relation_features(left, right) == {
        "shared_titles": [],
        "shared_query_entities": [],
        "shared_context_entities": [],
    }
