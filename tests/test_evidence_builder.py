from ladc.evidence_builder import build_evidence_graph, ingest_request_context
from ladc.workloads import WorkloadRequest


def _request() -> WorkloadRequest:
    return WorkloadRequest(
        request_id="q1",
        prompt="unused prompt",
        question="Who directed Blue Horizon?",
        context=(
            "[Film] Alice starred in Blue Horizon. Blue Horizon was directed by Bob Stone.\n"
            "[City] Blue Horizon premiered in Paris."
        ),
        # These deliberately disagree with the context.  Runtime ingestion
        # must not inspect either evaluation-only field.
        references=("Wrong answer",),
        supporting_facts=(("Wrong document", 99),),
        metadata={"dataset": "toy", "context_titles": ("Film", "City")},
    )


def test_ingest_splits_titled_context_and_keeps_runtime_status_uncertain():
    graph, sentence_ids = ingest_request_context(_request())

    assert len(sentence_ids) == 3
    sentences = [graph.nodes[node_id] for node_id in sentence_ids]
    assert all(node.status == "uncertain" for node in sentences)
    assert all(node.source_document_id and node.source_sentence_index is not None for node in sentences)
    assert any(node.text == "Blue Horizon was directed by Bob Stone." for node in sentences)
    assert not any(node.text == "Wrong answer" for node in graph.nodes.values())


def test_ingested_entities_enable_cross_sentence_graph_search():
    graph = build_evidence_graph([_request()])

    hits = graph.search("What happened in Paris?", query_entities=("Paris",))

    assert hits
    assert hits[0].source_document_id is not None
    assert "premiered in Paris" in hits[0].text


def test_same_source_context_is_reused_across_turns():
    first = _request()
    second = WorkloadRequest(
        request_id="q2",
        prompt="unused prompt",
        question="Which film did Alice star in?",
        context=first.context,
        references=(),
        metadata=first.metadata,
    )

    graph = build_evidence_graph([first, second])

    document_nodes = [node for node in graph.nodes.values() if node.node_type == "document"]
    assert len(document_nodes) == 2
    assert len([node for node in graph.nodes.values() if node.node_type == "question"]) == 2
