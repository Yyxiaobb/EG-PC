"""Build and inspect a small empirical quality calibration table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.calibration import EmpiricalQualityPredictor, collect_calibration_examples
from ladc.compressors import FixedBudgetCompressor
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import build_synthetic_request


def main() -> None:
    tokenizer = WhitespaceTokenizer()
    requests = [
        build_synthetic_request(
            f"calibration_{position}_{redundancy}",
            target_tokens=1024,
            tokenizer=tokenizer,
            answer_position=position,
            redundancy=redundancy,
            seed=100 + index,
        )
        for index, (position, redundancy) in enumerate(
            [("early", 0.2), ("early", 0.8), ("middle", 0.2), ("middle", 0.8), ("late", 0.2), ("late", 0.8)]
        )
    ]
    examples = collect_calibration_examples(
        requests,
        compressor=FixedBudgetCompressor(retain_ratio=0.5),
        tokenizer=tokenizer,
        retain_ratios=[1.0, 0.75, 0.5, 0.25],
    )
    predictor = EmpiricalQualityPredictor(safety_margin=0.02).fit(examples)
    output = {
        "examples": [example.__dict__ for example in examples],
        "sample_predictions": [
            {
                "request_id": request.request_id,
                "answer_position": request.answer_position,
                "predicted_quality_at_50_percent": predictor.predict(
                    request.prompt,
                    tokenizer=tokenizer,
                    retain_ratio=0.5,
                ),
            }
            for request in requests
        ],
    }
    Path("results/quality_calibration.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {len(examples)} calibration examples")


if __name__ == "__main__":
    main()
