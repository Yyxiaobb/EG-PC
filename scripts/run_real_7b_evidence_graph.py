"""Compare graph-aware compression with fixed-budget baselines on a real 7B model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.compressors import EvidenceGraphCompressor, FixedBudgetCompressor, NoCompressionCompressor
from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique, select_balanced_requests
from ladc.evidence_graph import EvidenceGraph
from ladc.experiments.runner import run_request
from ladc.inference import LlamaServerInferenceBackend
from ladc.quality import evaluate_real_request
from ladc.tokenizers import HuggingFaceTokenizer


def _load_requests(data_root: Path, tokenizer, max_per_bucket: int, seed: int):
    specs = [
        ("hotpotqa", ("hotpotqa", "hotpot_qa"), "validation", load_hotpotqa),
        ("2wikimultihopqa", ("2wikimultihopqa", "2WikiMultihopQA"), "validation", load_2wikimultihopqa),
        ("musique", ("musique", "MuSiQue"), "dev", load_musique),
    ]
    selected = []
    dataset_counts: dict[str, int] = {}
    for dataset, names, split, loader in specs:
        path = next((data_root / name for name in names if (data_root / name).exists()), None)
        if path is None:
            continue
        requests = loader(path, split=split)
        chosen = select_balanced_requests(
            requests,
            tokenizer=tokenizer,
            max_per_bucket=max_per_bucket,
            seed=seed,
        )
        selected.extend(chosen)
        dataset_counts[dataset] = len(chosen)
    return selected, dataset_counts


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _summarize(rows: list[dict[str, object]], baseline_scores: dict[str, float]) -> list[dict[str, object]]:
    summaries = []
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        quality_rows = [row for row in method_rows if row.get("quality_score") is not None]
        quality = [float(row["quality_score"]) for row in quality_rows]
        latency = [float(row["latency"]["end_to_end_ms"]) for row in method_rows]
        reductions = [float(row["token_reduction_rate"]) for row in method_rows]
        drops = [
            baseline_scores[str(row["request_id"])] - float(row["quality_score"])
            for row in quality_rows
            if str(row["method"]) != "no_compression"
            and str(row["request_id"]) in baseline_scores
        ]
        metrics = [row.get("quality_metrics", {}) for row in method_rows]
        compression_metadata = [row.get("metadata", {}).get("compression_metadata", {}) for row in method_rows]
        summaries.append(
            {
                "method": method,
                "n": len(method_rows),
                "mean_input_tokens": sum(float(row["compressed_prompt_tokens"]) for row in method_rows) / len(method_rows),
                "mean_token_reduction_rate": sum(reductions) / len(reductions),
                "mean_quality_score": sum(quality) / len(quality) if quality else 0.0,
                "mean_quality_drop_vs_no_compression": sum(drops) / len(drops) if drops else 0.0,
                "mean_exact_match": sum(float(metric.get("exact_match", 0.0)) for metric in metrics) / len(metrics),
                "mean_token_f1": sum(float(metric.get("token_f1", 0.0)) for metric in metrics) / len(metrics),
                "mean_answer_context_support": sum(float(metric.get("answer_context_support", 0.0)) for metric in metrics) / len(metrics),
                "mean_supporting_fact_precision": sum(float(metric.get("supporting_fact_precision", 0.0)) for metric in metrics) / len(metrics),
                "mean_supporting_fact_recall": sum(float(metric.get("supporting_fact_recall", 0.0)) for metric in metrics) / len(metrics),
                "mean_evidence_f1": sum(float(metric.get("evidence_f1", 0.0)) for metric in metrics) / len(metrics),
                "mean_end_to_end_ms": sum(latency) / len(latency),
                "p50_end_to_end_ms": _percentile(latency, 0.50),
                "p95_end_to_end_ms": _percentile(latency, 0.95),
                "mean_compression_ms": sum(float(row["latency"]["compression_ms"]) for row in method_rows) / len(method_rows),
                "mean_prefill_ms": sum(float(row["latency"]["prefill_ms"]) for row in method_rows) / len(method_rows),
                "mean_decode_ms": sum(float(row["latency"]["decode_ms"]) for row in method_rows) / len(method_rows),
                "mean_reused_evidence_count": sum(float(meta.get("reused_evidence_count", 0.0)) for meta in compression_metadata) / len(compression_metadata),
                "mean_historical_candidate_count": sum(float(meta.get("historical_candidate_count", 0.0)) for meta in compression_metadata) / len(compression_metadata),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument("--server", type=Path, default=Path("artifacts/llama_cpp/runtime/llama-server.exe"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(r"D:\HF_dl\model\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/real_7b_evidence_graph.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/real_7b_evidence_graph_summary.json"))
    parser.add_argument("--max-per-bucket", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9932)
    args = parser.parse_args()

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    requests, dataset_counts = _load_requests(args.data_root, tokenizer, args.max_per_bucket, seed=42)
    if not requests:
        raise RuntimeError(f"no evaluable requests found below {args.data_root}")

    backend = LlamaServerInferenceBackend(
        server_path=args.server,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path="results/llama_server_evidence_graph.log",
    )
    graph_compressor = EvidenceGraphCompressor(graph=EvidenceGraph(), name="evidence_graph_dynamic")
    compressors = {
        "no_compression": NoCompressionCompressor(),
        "fixed_budget_50": FixedBudgetCompressor(retain_ratio=0.50, name="fixed_budget_50"),
        "evidence_graph_dynamic": graph_compressor,
    }
    rows: list[dict[str, object]] = []
    baseline_scores: dict[str, float] = {}
    try:
        with backend:
            backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
            for request in requests:
                baseline = run_request(
                    request,
                    compressor=compressors["no_compression"],
                    backend=backend,
                    tokenizer=tokenizer,
                    quality_fn=evaluate_real_request,
                )
                baseline_scores[request.request_id] = baseline.quality_score or 0.0
                rows.append(baseline.to_dict())
                for method in ("fixed_budget_50", "evidence_graph_dynamic"):
                    result = run_request(
                        request,
                        compressor=compressors[method],
                        backend=backend,
                        tokenizer=tokenizer,
                        baseline_quality_score=baseline.quality_score,
                        quality_fn=evaluate_real_request,
                    )
                    rows.append(result.to_dict())
    finally:
        backend.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "dataset_counts": dataset_counts,
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "summaries": _summarize(rows, baseline_scores),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
