import json
from pathlib import Path

from ladc.evidence_graph import EvidenceGraph, EvidenceSpan


def _graph() -> EvidenceGraph:
    graph = EvidenceGraph()
    graph.add_turn(
        "turn-1",
        "Which director made the film starring Alice?",
        [
            EvidenceSpan(
                document_id="doc-1",
                sentence_index=0,
                title="Film",
                text="Alice starred in the film Blue Horizon.",
                entities=("Alice", "Blue Horizon"),
                claims=("Alice starred in Blue Horizon",),
            ),
            EvidenceSpan(
                document_id="doc-1",
                sentence_index=1,
                title="Film",
                text="Blue Horizon was directed by Bob Stone.",
                entities=("Blue Horizon", "Bob Stone"),
                claims=("Blue Horizon was directed by Bob Stone",),
            ),
        ],
        answer="Bob Stone",
        answer_confidence=0.9,
    )
    return graph


def test_add_turn_keeps_source_backed_evidence_and_does_not_rank_answer_node():
    graph = _graph()

    hits = graph.search("Who directed Blue Horizon?", query_entities=("Blue Horizon",))

    assert hits
    assert hits[0].source_document_id == "doc-1"
    assert hits[0].source_sentence_index == 1
    assert "Bob Stone" in hits[0].text
    assert all(graph.nodes[hit.node_id].node_type == "sentence" for hit in hits)


def test_graph_bridge_recovers_sentence_with_weak_direct_query_overlap():
    graph = _graph()

    hits = graph.search("What happened to Bob Stone?", query_entities=("Bob Stone",), top_k=5)

    texts = {hit.text for hit in hits}
    assert "Blue Horizon was directed by Bob Stone." in texts
    assert any(hit.hops >= 1 for hit in hits)


def test_conflicted_evidence_is_not_used_by_default():
    graph = EvidenceGraph()
    graph.add_span(
        EvidenceSpan(
            document_id="doc-conflict",
            sentence_index=0,
            text="The event happened in Paris.",
            entities=("event", "Paris"),
            status="conflicted",
        )
    )

    assert graph.search("Where did the event happen?") == []
    assert graph.search("Where did the event happen?", allowed_statuses=("conflicted",))


def test_graph_round_trip_preserves_nodes_edges_and_search():
    graph = _graph()
    path = Path("results") / "_test_evidence_graph.json"

    try:
        graph.save(path)
        loaded = EvidenceGraph.load(path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert len(loaded.nodes) == len(graph.nodes)
        assert len(loaded.edges) == len(graph.edges)
        assert loaded.search("Who directed Blue Horizon?", query_entities=("Blue Horizon",))
    finally:
        path.unlink(missing_ok=True)
