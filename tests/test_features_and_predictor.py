import pytest

from ladc.features import extract_prompt_features
from ladc.predictor import HeuristicQualityPredictor
from ladc.tokenizers import WhitespaceTokenizer


def test_prompt_features_are_available_without_gold_labels():
    tokenizer = WhitespaceTokenizer()
    features = extract_prompt_features(
        "context evidence context evidence Question: which evidence?",
        tokenizer=tokenizer,
    )
    assert features.prompt_tokens == 7
    assert features.redundancy_ratio > 0.0
    assert features.question_tokens == 2
    assert features.context_query_overlap > 0.0
    assert 0.0 <= features.query_alignment_position <= 1.0


def test_predictor_quality_decreases_with_more_aggressive_compression():
    tokenizer = WhitespaceTokenizer()
    predictor = HeuristicQualityPredictor()
    light = predictor.predict("one two three Question: answer", tokenizer=tokenizer, retain_ratio=0.75)
    aggressive = predictor.predict("one two three Question: answer", tokenizer=tokenizer, retain_ratio=0.25)
    assert 0.0 <= aggressive < light <= 1.0


def test_predictor_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        HeuristicQualityPredictor().predict_from_features(
            extract_prompt_features("one", tokenizer=WhitespaceTokenizer()),
            0.0,
        )
