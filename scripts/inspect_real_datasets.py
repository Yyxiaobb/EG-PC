"""Inspect a few local dataset rows without modifying source data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow this script to run from a fresh checkout before ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique
from ladc.tokenizers import WhitespaceTokenizer


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=2)
    args = parser.parse_args()
    loaders = {
        "hotpotqa": load_hotpotqa,
        "2wikimultihopqa": load_2wikimultihopqa,
        "musique": load_musique,
    }
    dataset_directories = {
        "hotpotqa": (args.root / "hotpotqa", args.root / "hotpot_qa"),
        "2wikimultihopqa": (args.root / "2wikimultihopqa", args.root / "2WikiMultihopQA"),
        "musique": (args.root / "musique", args.root / "MuSiQue"),
    }
    dataset_splits = {
        "hotpotqa": "validation",
        "2wikimultihopqa": "validation",
        "musique": "dev",
    }
    tokenizer = WhitespaceTokenizer()
    report: dict[str, object] = {}
    for name, loader in loaders.items():
        dataset_path = next((candidate for candidate in dataset_directories[name] if candidate.exists()), None)
        if dataset_path is None:
            report[name] = {"status": "missing", "paths": [str(item) for item in dataset_directories[name]]}
            continue
        try:
            requests = loader(
                dataset_path,
                split=dataset_splits[name],
                max_examples=args.max_examples,
            )
            report[name] = {
                "status": "ok",
                "split": dataset_splits[name],
                "count": len(requests),
                "samples": [
                    {
                        "request_id": request.request_id,
                        "prompt_tokens_whitespace": tokenizer.count_tokens(request.prompt),
                        "reference_count": len(request.references),
                        "supporting_fact_count": len(request.supporting_facts),
                        "metadata": request.metadata,
                    }
                    for request in requests
                ],
            }
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            report[name] = {"status": "error", "error": str(exc)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
