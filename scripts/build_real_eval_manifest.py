"""Build a reproducible, annotation-light manifest for real-model evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.datasets import build_length_manifest, select_balanced_requests
from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique
from ladc.tokenizers import HuggingFaceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Local Transformers tokenizer/model directory")
    parser.add_argument("--tokenizer-model-id", default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = HuggingFaceTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    specs = [
        ("hotpotqa", ("hotpotqa", "hotpot_qa"), "validation", load_hotpotqa),
        ("2wikimultihopqa", ("2wikimultihopqa", "2WikiMultihopQA"), "validation", load_2wikimultihopqa),
        ("musique", ("musique", "MuSiQue"), "dev", load_musique),
    ]
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for dataset, directory_names, split, loader in specs:
        dataset_path = next((args.data_root / name for name in directory_names if (args.data_root / name).exists()), None)
        if dataset_path is None:
            summary[dataset] = {"status": "missing"}
            continue
        requests = loader(dataset_path, split=split)
        selected = select_balanced_requests(
            requests,
            tokenizer=tokenizer,
            max_per_bucket=args.max_per_bucket,
            seed=args.seed,
        )
        selected_rows = build_length_manifest(selected, tokenizer=tokenizer)
        rows.extend(selected_rows)
        summary[dataset] = {
            "status": "ok",
            "split": split,
            "loaded": len(requests),
            "selected": len(selected_rows),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "max_per_bucket": args.max_per_bucket,
                "tokenizer_model_id": args.tokenizer_model_id,
                "summary": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
