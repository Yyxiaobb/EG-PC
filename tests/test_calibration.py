from ladc.calibration import EmpiricalQualityPredictor, collect_calibration_examples
from ladc.compressors import FixedBudgetCompressor
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import build_synthetic_request


def test_calibration_collects_observed_quality():
    tokenizer = WhitespaceTokenizer()
    request = build_synthetic_request(
        "early",
        target_tokens=512,
        tokenizer=tokenizer,
        answer_position="early",
    )
    examples = collect_calibration_examples(
        [request],
        compressor=FixedBudgetCompressor(retain_ratio=0.5),
        tokenizer=tokenizer,
        retain_ratios=[1.0, 0.5],
    )
    assert len(examples) == 2
    assert examples[0].observed_quality == 1.0
    assert 0.0 <= examples[0].query_alignment_position <= 1.0


def test_empirical_predictor_can_fit_and_predict():
    tokenizer = WhitespaceTokenizer()
    request = build_synthetic_request("r1", target_tokens=512, tokenizer=tokenizer)
    examples = collect_calibration_examples(
        [request],
        compressor=FixedBudgetCompressor(retain_ratio=0.5),
        tokenizer=tokenizer,
        retain_ratios=[1.0, 0.5],
    )
    predictor = EmpiricalQualityPredictor().fit(examples)
    prediction = predictor.predict(request.prompt, tokenizer=tokenizer, retain_ratio=0.5)
    assert 0.0 <= prediction <= 1.0
