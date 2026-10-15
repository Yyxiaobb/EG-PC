"""Compression-only pilot on a deterministic fraction of each QA dataset.

The expensive generation model is intentionally excluded from this first
LLMLingua-2 validation.  The report measures prompt tokens, compressor cost,
and source-backed evidence retention on exactly 10% of each local split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.compressors import (
    EvidenceGraphCompressor,
    FixedBudgetCompressor,
    LLMLingua2Compressor,
    NoCompressionCompressor,
)
from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique
from ladc.evidence_graph import EvidenceGraph
from ladc.quality import _supporting_fact_metrics
from ladc.tokenizers import HuggingFaceTokenizer


def _sample(requests, fraction: float, seed: int):
    count = max(1, math.ceil(len(requests) * fraction))
    return random.Random(seed).sample(requests, count)


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for method in sorted({str(row["method"]) for row in rows}):
        items = [row for row in rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "n": len(items),
                "mean_input_tokens": sum(float(row["compressed_prompt_tokens"]) for row in items) / len(items),
                "mean_token_reduction_rate": sum(float(row["token_reduction_rate"]) for row in items) / len(items),
                "mean_compression_ms": sum(float(row["compression_ms"]) for row in items) / len(items),
                "p50_compression_ms": sorted(float(row["compression_ms"]) for row in items)[len(items) // 2],
                "mean_evidence_f1": sum(float(row["evidence_f1"]) for row in items) / len(items),
                "mean_supporting_fact_recall": sum(float(row["supporting_fact_recall"]) for row in items) / len(items),
                "mean_answer_presence_in_context": sum(float(row["answer_presence_in_context"]) for row in items) / len(items),
            }
        )
    return summaries


def _summarize_by_dataset(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for summary in _summarize([row for row in rows if row["dataset"] == dataset]):
            output.append({"dataset": dataset, **summary})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument(
        "--llmlingua2-model",
        type=Path,
        default=Path(r"D:\HF_dl\model\llmlingua-2-bert-base-multilingual-cased-meetingbank"),
    )
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/llmlingua2_fraction_10pct.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/llmlingua2_fraction_10pct_summary.json"))
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    specs = [
        ("hotpotqa", ("hotpotqa", "hotpot_qa"), "validation", load_hotpotqa),
        ("2wikimultihopqa", ("2wikimultihopqa", "2WikiMultihopQA"), "validation", load_2wikimultihopqa),
        ("musique", ("musique", "MuSiQue"), "dev", load_musique),
    ]
    selected = []
    dataset_counts: dict[str, dict[str, int]] = {}
    for dataset, names, split, loader in specs:
        path = next((args.data_root / name for name in names if (args.data_root / name).exists()), None)
        if path is None:
            continue
        all_requests = loader(path, split=split)
        chosen = _sample(all_requests, args.fraction, args.seed)
        selected.extend((dataset, request) for request in chosen)
        dataset_counts[dataset] = {"total": len(all_requests), "sampled": len(chosen)}

    if not selected:
        raise RuntimeError("no dataset rows found")

    compressors = {
        "no_compression": NoCompressionCompressor(),
        "fixed_budget_50": FixedBudgetCompressor(retain_ratio=0.50, name="fixed_budget_50"),
        "evidence_graph": EvidenceGraphCompressor(graph=EvidenceGraph(), name="evidence_graph"),
        "llmlingua2_50": LLMLingua2Compressor(
            str(args.llmlingua2_model),
            retain_ratio=0.50,
            device_map="cuda",
            model_config={"local_files_only": True},
        ),
    }
    rows: list[dict[str, object]] = []
    for dataset, request in selected:
        for method, compressor in compressors.items():
            request_compressor = getattr(compressor, "compress_request", None)
            result = (
                request_compressor(request, tokenizer=tokenizer)
                if callable(request_compressor)
                else compressor.compress(request.prompt, tokenizer=tokenizer)
            )
            evidence = _supporting_fact_metrics(request, result.compressed_prompt)
            reference = request.references[0] if request.references else ""
            normalized_prompt = result.compressed_prompt.casefold()
            rows.append(
                {
                    "dataset": dataset,
                    "request_id": request.request_id,
                    "method": method,
                    "original_prompt_tokens": result.original_prompt_tokens,
                    "compressed_prompt_tokens": result.compressed_prompt_tokens,
                    "token_reduction_rate": result.token_reduction_rate,
                    "compression_ms": result.latency_ms,
                    "evidence_f1": evidence["evidence_f1"],
                    "supporting_fact_recall": evidence["supporting_fact_recall"],
                    "answer_presence_in_context": float(bool(reference) and reference.casefold() in normalized_prompt),
                    "metadata": result.metadata,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "fraction": args.fraction,
        "seed": args.seed,
        "dataset_counts": dataset_counts,
        "llmlingua2_model": str(args.llmlingua2_model),
        "rows": len(rows),
        "summaries": _summarize(rows),
        "summaries_by_dataset": _summarize_by_dataset(rows),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
