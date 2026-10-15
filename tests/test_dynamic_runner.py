from ladc.compressors import FixedBudgetCompressor, NoCompressionCompressor
from ladc.controller import BudgetCandidate, DynamicBudgetController
from ladc.experiments.runner import run_dynamic_request
from ladc.inference import SimulatedInferenceBackend
from ladc.quality import evaluate_context_preservation
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import build_synthetic_request


def test_dynamic_runner_executes_only_selected_plan():
    tokenizer = WhitespaceTokenizer()
    request = build_synthetic_request("r1", target_tokens=512, tokenizer=tokenizer)
    row, decision = run_dynamic_request(
        request,
        candidates=[
            BudgetCandidate("no_compression", 1.0, 1.0, 0.0),
            BudgetCandidate("fixed_budget", 0.5, 0.99, 0.5),
        ],
        compressors={
            "no_compression": NoCompressionCompressor(),
            "fixed_budget": FixedBudgetCompressor(retain_ratio=0.5),
        },
        controller=DynamicBudgetController(quality_drop_tolerance=0.03),
        backend=SimulatedInferenceBackend(),
        tokenizer=tokenizer,
        quality_fn=lambda sample, prompt_text, output: evaluate_context_preservation(
            prompt_text, sample.expected_answer
        ),
    )
    assert decision.candidate.method == "fixed_budget"
    assert row.method == "dynamic:fixed_budget"
    assert "budget_decision" in row.metadata
