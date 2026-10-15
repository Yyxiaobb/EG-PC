"""Run 7B quality and end-to-end latency with session-scoped graph memory."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.compressors import (
    FixedBudgetCompressor,
    LLMLingua2Compressor,
    NoCompressionCompressor,
    SessionEvidenceGraphCompressor,
)
from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique
from ladc.experiments.runner import run_request
from ladc.inference import LlamaServerInferenceBackend
from ladc.quality import evaluate_real_request
from ladc.tokenizers import HuggingFaceTokenizer


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))]


def _load_fraction(data_root: Path, fraction: float, tokenizer, seed: int):
    specs = [
        ("hotpotqa", ("hotpotqa", "hotpot_qa"), "validation", load_hotpotqa),
        ("2wikimultihopqa", ("2wikimultihopqa", "2WikiMultihopQA"), "validation", load_2wikimultihopqa),
        ("musique", ("musique", "MuSiQue"), "dev", load_musique),
    ]
    selected = []
    counts = {}
    rng = random.Random(seed)
    for dataset, names, split, loader in specs:
        path = next((data_root / name for name in names if (data_root / name).exists()), None)
        if path is None:
            continue
        requests = loader(path, split=split)
        count = max(1, math.ceil(len(requests) * fraction))
        chosen = rng.sample(requests, count)
        counts[dataset] = {"total": len(requests), "sampled": len(chosen)}
        for request in chosen:
            # Independent benchmark questions are independent application
            # sessions unless an upstream application supplies a family ID.
            metadata = {**request.metadata, "session_id": f"{dataset}:{request.request_id}"}
            selected.append((dataset, replace(request, metadata=metadata)))
    return selected, counts


def _summarize(rows: list[dict[str, object]], baseline: dict[str, float]):
    result = []
    for method in sorted({str(row["method"]) for row in rows}):
        items = [row for row in rows if row["method"] == method]
        quality = [float(row["quality_score"]) for row in items if row["quality_score"] is not None]
        latency = [float(row["latency"]["end_to_end_ms"]) for row in items]
        drops = [baseline[str(row["request_id"])] - float(row["quality_score"])
                 for row in items if row["quality_score"] is not None and method != "no_compression"]
        result.append({
            "method": method,
            "n": len(items),
            "mean_input_tokens": sum(float(row["compressed_prompt_tokens"]) for row in items) / len(items),
            "mean_token_reduction_rate": sum(float(row["token_reduction_rate"]) for row in items) / len(items),
            "mean_quality_score": sum(quality) / len(quality) if quality else 0.0,
            "mean_quality_drop_vs_no_compression": sum(drops) / len(drops) if drops else 0.0,
            "mean_exact_match": sum(float(row["quality_metrics"].get("exact_match", 0.0)) for row in items) / len(items),
            "mean_token_f1": sum(float(row["quality_metrics"].get("token_f1", 0.0)) for row in items) / len(items),
            "mean_evidence_f1": sum(float(row["quality_metrics"].get("evidence_f1", 0.0)) for row in items) / len(items),
            "mean_compression_ms": sum(float(row["latency"]["compression_ms"]) for row in items) / len(items),
            "mean_prefill_ms": sum(float(row["latency"]["prefill_ms"]) for row in items) / len(items),
            "mean_decode_ms": sum(float(row["latency"]["decode_ms"]) for row in items) / len(items),
            "mean_end_to_end_ms": sum(latency) / len(latency),
            "p50_end_to_end_ms": _percentile(latency, 0.50),
            "p95_end_to_end_ms": _percentile(latency, 0.95),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument("--server", type=Path, default=Path("artifacts/llama_cpp/runtime/llama-server.exe"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"))
    parser.add_argument("--llmlingua2-model", type=Path, default=Path(r"D:\HF_dl\model\llmlingua-2-bert-base-multilingual-cased-meetingbank"))
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9933)
    parser.add_argument("--output", type=Path, default=Path("results/real_7b_session_llmlingua2.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/real_7b_session_llmlingua2_summary.json"))
    args = parser.parse_args()

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    requests, dataset_counts = _load_fraction(args.data_root, args.fraction, tokenizer, args.seed)
    if not requests:
        raise RuntimeError("no evaluable requests found")

    compressors = {
        "no_compression": NoCompressionCompressor(),
        "fixed_budget_50": FixedBudgetCompressor(retain_ratio=0.50, name="fixed_budget_50"),
        "evidence_graph_session": SessionEvidenceGraphCompressor(),
        "llmlingua2_50": LLMLingua2Compressor(
            str(args.llmlingua2_model),
            retain_ratio=0.50,
            device_map="cuda",
            model_config={"local_files_only": True},
        ),
    }
    backend = LlamaServerInferenceBackend(
        server_path=args.server,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path="results/llama_server_session_llmlingua2.log",
    )
    rows = []
    baseline_scores: dict[str, float] = {}
    try:
        with backend:
            backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
            for dataset, request in requests:
                baseline = run_request(request, compressor=compressors["no_compression"], backend=backend, tokenizer=tokenizer, quality_fn=evaluate_real_request)
                baseline_scores[request.request_id] = baseline.quality_score or 0.0
                rows.append({**baseline.to_dict(), "dataset": dataset})
                for method in ("fixed_budget_50", "evidence_graph_session", "llmlingua2_50"):
                    record = run_request(request, compressor=compressors[method], backend=backend, tokenizer=tokenizer, baseline_quality_score=baseline.quality_score, quality_fn=evaluate_real_request)
                    rows.append({**record.to_dict(), "dataset": dataset})
    finally:
        backend.close()
        compressors["evidence_graph_session"].close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "fraction": args.fraction,
        "dataset_counts": dataset_counts,
        "model": str(args.model),
        "llmlingua2_model": str(args.llmlingua2_model),
        "rows": len(rows),
        "summaries": _summarize(rows, baseline_scores),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
