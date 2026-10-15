import json
from pathlib import Path

from ladc.datasets import (
    build_length_manifest,
    length_bucket,
    load_2wikimultihopqa,
    load_hotpotqa,
    load_musique,
    select_balanced_requests,
)
from ladc.workloads import WorkloadRequest


def test_musique_loader_separates_evaluation_annotations():
    path = Path("results") / "_test_musique_train.jsonl"
    row = {
        "id": "m1",
        "question": "Who wrote the book?",
        "answer": "A. Author",
        "answer_aliases": ["Author A"],
        "answerable": True,
        "paragraphs": [
            {"title": "Book", "paragraph_text": "The book was published in 2001.", "idx": 0, "is_supporting": False},
            {"title": "Author", "paragraph_text": "A. Author wrote the book.", "idx": 1, "is_supporting": True},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    try:
        request = load_musique(path, max_examples=1)[0]

        assert request.request_id == "m1"
        assert request.references == ("A. Author", "Author A")
        assert request.supporting_facts == (("Author", 1),)
        assert request.question in request.prompt
        assert request.metadata["dataset"] == "musique"
        assert "references" not in request.metadata
        assert "supporting_facts" not in request.metadata
    finally:
        path.unlink(missing_ok=True)


def test_hotpotqa_loader_formats_context_and_honors_limit():
    path = Path("results") / "_test_hotpot_dev.jsonl"
    rows = [
        {
            "id": "h1",
            "question": "What is the answer?",
            "answer": "Paris",
            "context": [["France", ["Paris is the capital."]]],
            "supporting_facts": [["France", 0]],
        },
        {
            "id": "h2",
            "question": "Another question",
            "answer": "Rome",
            "context": [["Italy", ["Rome is the capital."]]],
            "supporting_facts": [["Italy", 0]],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    try:
        requests = load_hotpotqa(path, max_examples=1)

        assert len(requests) == 1
        assert "[France] Paris is the capital." in requests[0].context
        assert requests[0].supporting_facts == (("France", 0),)
    finally:
        path.unlink(missing_ok=True)


def test_2wiki_loader_handles_columnar_context_and_facts():
    path = Path("results") / "_test_2wiki.jsonl"
    row = {
        "id": "w1",
        "question": "Which person?",
        "answer": "Alice",
        "evidences": [],
        "context": {
            "title": ["First", "Second"],
            "sentences": [["Alice is here."], ["Bob is there."]],
        },
        "supporting_facts": {"title": ["First"], "sent_id": [0]},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    try:
        request = load_2wikimultihopqa(path, max_examples=1)[0]

        assert request.metadata["context_document_count"] == 2
        assert "[First] Alice is here." in request.context
        assert request.supporting_facts == (("First", 0),)
    finally:
        path.unlink(missing_ok=True)


def test_length_bucket_boundaries():
    assert length_bucket(1024) == "<2K"
    assert length_bucket(2048) == "2K-4K"
    assert length_bucket(4096) == "4K-8K"
    assert length_bucket(8192) == "8K-16K"
    assert length_bucket(16384) == "16K+"


class CharacterTokenizer:
    def count_tokens(self, text):
        return len(text)

    def truncate_to_tokens(self, text, max_tokens):
        return text[:max_tokens]


def test_native_length_manifest_and_balanced_sampling_are_reproducible():
    tokenizer = CharacterTokenizer()
    requests = [
        WorkloadRequest(
            request_id=f"r{index}",
            prompt="x" * length,
            question="q",
            context="x" * length,
            references=("answer",),
            metadata={"dataset": "toy", "split": "validation"},
        )
        for index, length in enumerate((100, 2200, 2300, 5000))
    ]

    manifest = build_length_manifest(requests, tokenizer=tokenizer)
    selected_a = select_balanced_requests(requests, tokenizer=tokenizer, max_per_bucket=1, seed=7)
    selected_b = select_balanced_requests(requests, tokenizer=tokenizer, max_per_bucket=1, seed=7)

    assert manifest[1]["prompt_tokens"] == 2200
    assert manifest[1]["length_bucket"] == "2K-4K"
    assert [item.request_id for item in selected_a] == [item.request_id for item in selected_b]
