"""Run the initial heuristic dynamic budget policy on synthetic requests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.analysis import summarize_records
from ladc.compressors import FixedBudgetCompressor, NoCompressionCompressor
from ladc.controller import BudgetCandidate, DynamicBudgetController
from ladc.experiments.runner import run_dynamic_request
from ladc.inference import SimulatedInferenceBackend
from ladc.predictor import HeuristicQualityPredictor
from ladc.quality import evaluate_context_preservation
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import build_length_matrix


def main() -> None:
    tokenizer = WhitespaceTokenizer()
    predictor = HeuristicQualityPredictor()
    controller = DynamicBudgetController(
        prefill_ms_per_token=0.08,
        decode_ms_per_token=2.0,
        quality_drop_tolerance=0.03,
    )
    backend = SimulatedInferenceBackend()
    compressors = {
        "no_compression": NoCompressionCompressor(),
        "fixed_budget": FixedBudgetCompressor(retain_ratio=0.5),
    }
    rows = []
    decisions = []
    for request in build_length_matrix([2_048, 4_096, 8_192, 16_384], tokenizer=tokenizer):
        ratios = [1.0, 0.75, 0.50, 0.25]
        candidates = [
            BudgetCandidate(
                method="no_compression" if ratio == 1.0 else "fixed_budget",
                target_budget_ratio=ratio,
                predicted_quality=predictor.predict(request.prompt, tokenizer=tokenizer, retain_ratio=ratio),
                predicted_compression_ms=0.0 if ratio == 1.0 else 0.5,
            )
            for ratio in ratios
        ]
        row, decision = run_dynamic_request(
            request,
            candidates=candidates,
            compressors=compressors,
            controller=controller,
            backend=backend,
            tokenizer=tokenizer,
            quality_fn=lambda sample, prompt_text, output: evaluate_context_preservation(
                prompt_text, sample.expected_answer
            ),
        )
        rows.append(row)
        decisions.append(decision.to_dict())

    output = Path("results/dynamic_synthetic.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row.to_dict(), ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    Path("results/dynamic_synthetic_summary.json").write_text(
        json.dumps({"summary": summarize_records(rows), "decisions": decisions}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} dynamic rows")


if __name__ == "__main__":
    main()
