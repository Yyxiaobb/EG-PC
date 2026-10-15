"""Run a small synthetic matrix and write JSONL results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution from a clean checkout before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.compressors import FixedBudgetCompressor, NoCompressionCompressor, TruncationCompressor
from ladc.analysis import summarize_records
from ladc.experiments import run_workload_matrix
from ladc.inference import SimulatedInferenceBackend
from ladc.quality import evaluate_context_preservation
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import build_length_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/synthetic_smoke.jsonl")
    args = parser.parse_args()

    tokenizer = WhitespaceTokenizer()
    requests = build_length_matrix([2_048, 4_096, 8_192, 16_384], tokenizer=tokenizer)
    compressors = [
        NoCompressionCompressor(),
        FixedBudgetCompressor(retain_ratio=0.5),
        TruncationCompressor(max_tokens=2_048),
    ]
    backend = SimulatedInferenceBackend()
    rows = run_workload_matrix(
        requests,
        compressors=compressors,
        backend=backend,
        tokenizer=tokenizer,
        target_budget_ratio=None,
        quality_fn=lambda sample, prompt_text, output: evaluate_context_preservation(
            prompt_text, sample.expected_answer
        ),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=True) + "\n")
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summarize_records(rows), indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {output_path}")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
