import pytest

from ladc.controller import BudgetCandidate, DynamicBudgetController


def test_controller_selects_fastest_candidate_within_quality_floor():
    controller = DynamicBudgetController(
        prefill_ms_per_token=1.0,
        decode_ms_per_token=1.0,
        quality_drop_tolerance=0.03,
    )
    decision = controller.choose(
        original_prompt_tokens=1000,
        output_tokens=10,
        candidates=[
            BudgetCandidate("no_compression", 1.0, 1.0, 0.0),
            BudgetCandidate("fixed_budget", 0.5, 0.99, 1.0),
            BudgetCandidate("fixed_budget", 0.25, 0.90, 1.0),
        ],
    )
    assert decision.candidate.target_budget_ratio == 0.5
    assert decision.quality_constraint_satisfied is True


def test_controller_falls_back_to_best_predicted_quality():
    controller = DynamicBudgetController(quality_drop_tolerance=0.01)
    decision = controller.choose(
        original_prompt_tokens=1000,
        output_tokens=10,
        candidates=[
            BudgetCandidate("fixed_budget", 0.25, 0.80, 1.0),
            BudgetCandidate("fixed_budget", 0.50, 0.81, 0.5),
        ],
        baseline_quality=1.0,
    )
    assert decision.quality_constraint_satisfied is False
    assert decision.candidate.predicted_quality == 0.81
    assert "highest predicted quality" in decision.reason


def test_controller_rejects_invalid_inputs():
    controller = DynamicBudgetController()
    with pytest.raises(ValueError):
        controller.choose(original_prompt_tokens=0, output_tokens=10, candidates=[])
