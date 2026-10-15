import pytest

from ladc.compressors import FixedBudgetCompressor, NoCompressionCompressor, TruncationCompressor
from ladc.tokenizers import WhitespaceTokenizer


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()


@pytest.fixture
def prompt():
    return " ".join(f"token_{index}" for index in range(100))


def test_no_compression_preserves_prompt(tokenizer, prompt):
    result = NoCompressionCompressor().compress(prompt, tokenizer=tokenizer)
    assert result.method == "no_compression"
    assert result.original_prompt_tokens == 100
    assert result.compressed_prompt_tokens == 100
    assert result.token_reduction_rate == 0.0


def test_fixed_budget_uses_fraction_of_each_request(tokenizer, prompt):
    result = FixedBudgetCompressor(retain_ratio=0.5).compress(prompt, tokenizer=tokenizer)
    assert result.compressed_prompt_tokens == 50
    assert result.token_reduction_rate == 0.5


def test_fixed_budget_accepts_distinct_experiment_name(tokenizer, prompt):
    result = FixedBudgetCompressor(retain_ratio=0.5, name="fixed_budget_50").compress(
        prompt,
        tokenizer=tokenizer,
    )
    assert result.method == "fixed_budget_50"


def test_truncation_uses_absolute_budget(tokenizer, prompt):
    result = TruncationCompressor(max_tokens=16).compress(prompt, tokenizer=tokenizer)
    assert result.compressed_prompt_tokens == 16
    assert result.compressed_prompt.split()[-1] == "token_15"


def test_runtime_budget_overrides_default(tokenizer, prompt):
    result = FixedBudgetCompressor(retain_ratio=0.8).compress(
        prompt,
        tokenizer=tokenizer,
        target_budget_ratio=0.25,
    )
    assert result.compressed_prompt_tokens == 25


def test_invalid_baseline_arguments_are_rejected():
    with pytest.raises(ValueError):
        FixedBudgetCompressor(retain_ratio=0.0)
    with pytest.raises(ValueError):
        TruncationCompressor(max_tokens=0)
