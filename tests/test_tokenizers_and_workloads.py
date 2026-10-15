import pytest

from ladc.tokenizers import HuggingFaceTokenizer, WhitespaceTokenizer
from ladc.workloads import build_length_matrix, build_synthetic_request


def test_whitespace_tokenizer_counts_and_truncates():
    tokenizer = WhitespaceTokenizer()
    assert tokenizer.count_tokens("one two three") == 3
    assert tokenizer.truncate_to_tokens("one two three", 2) == "one two"


def test_synthetic_request_hits_requested_length_without_exceeding_it():
    tokenizer = WhitespaceTokenizer()
    request = build_synthetic_request("r1", target_tokens=256, tokenizer=tokenizer)
    assert tokenizer.count_tokens(request.prompt) <= 256
    assert request.metadata["actual_tokens"] == tokenizer.count_tokens(request.prompt)


def test_synthetic_request_keeps_answer_for_non_truncation_tests():
    tokenizer = WhitespaceTokenizer()
    request = build_synthetic_request("r1", target_tokens=512, tokenizer=tokenizer)
    assert request.expected_answer in request.prompt
    assert request.metadata["answer_is_in_original_prompt"] is True


def test_length_matrix_is_deterministic_and_validates_small_lengths():
    tokenizer = WhitespaceTokenizer()
    matrix = build_length_matrix([256, 1024, 2048], tokenizer=tokenizer)
    assert [item.request_id for item in matrix] == [
        "synthetic_256",
        "synthetic_1024",
        "synthetic_2048",
    ]
    with pytest.raises(ValueError):
        build_synthetic_request("too_short", target_tokens=32, tokenizer=tokenizer)


class FakeNativeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text)))

    def decode(self, token_ids, **kwargs):
        assert kwargs["skip_special_tokens"] is False
        return "x" * len(token_ids)


def test_huggingface_adapter_uses_native_encode_and_decode():
    tokenizer = HuggingFaceTokenizer(FakeNativeTokenizer())

    assert tokenizer.count_tokens("abcd") == 4
    assert tokenizer.truncate_to_tokens("abcdef", 3) == "xxx"
    assert tokenizer.truncate_to_tokens("abcdef", 0) == ""
