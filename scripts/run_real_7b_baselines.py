"""Run real 7B GGUF compression baselines on a small public-data pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.compressors import FixedBudgetCompressor, NoCompressionCompressor
from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique, select_balanced_requests
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
    for dataset, names, split, loader in specs:
        path = next((data_root / name for name in names if (data_root / name).exists()), None)
        if path is None:
            continue
        requests = loader(path, split=split)
        selected.extend(
            select_balanced_requests(
                requests,
                tokenizer=tokenizer,
                max_per_bucket=max_per_bucket,
                seed=seed,
            )
        )
    return selected


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
    parser.add_argument("--output", type=Path, default=Path("results/real_7b_baselines.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/real_7b_baselines_summary.json"))
    parser.add_argument("--max-per-bucket", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9931)
    args = parser.parse_args()

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    requests = _load_requests(args.data_root, tokenizer, args.max_per_bucket, seed=42)
    if not requests:
        raise RuntimeError("no evaluable requests found")

    backend = LlamaServerInferenceBackend(
        server_path=args.server,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path="results/llama_server_7b.log",
    )
    compressors = {
        "no_compression": NoCompressionCompressor(),
        "fixed_budget_75": FixedBudgetCompressor(retain_ratio=0.75, name="fixed_budget_75"),
        "fixed_budget_50": FixedBudgetCompressor(retain_ratio=0.50, name="fixed_budget_50"),
        "fixed_budget_25": FixedBudgetCompressor(retain_ratio=0.25, name="fixed_budget_25"),
    }
    rows = []
    baseline_scores: dict[str, float] = {}
    try:
        with backend:
            # One request primes server-side CUDA kernels; it is excluded from results.
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
                for method in ("fixed_budget_75", "fixed_budget_50", "fixed_budget_25"):
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

    summaries = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        quality = [row["quality_score"] for row in method_rows if row["quality_score"] is not None]
        latency = [row["latency"]["end_to_end_ms"] for row in method_rows]
        token_reduction = [row["token_reduction_rate"] for row in method_rows]
        drops = [
            baseline_scores[row["request_id"]] - row["quality_score"]
            for row in method_rows
            if row["method"] != "no_compression" and row["quality_score"] is not None
        ]
        summaries.append(
            {
                "method": method,
                "n": len(method_rows),
                "mean_input_tokens": sum(row["compressed_prompt_tokens"] for row in method_rows) / len(method_rows),
                "mean_token_reduction_rate": sum(token_reduction) / len(token_reduction),
                "mean_end_to_end_ms": sum(latency) / len(latency),
                "p50_end_to_end_ms": sorted(latency)[len(latency) // 2],
                "mean_quality_score": sum(quality) / len(quality) if quality else 0.0,
                "mean_quality_drop_vs_no_compression": sum(drops) / len(drops) if drops else 0.0,
                "mean_exact_match": sum(row["quality_metrics"].get("exact_match", 0.0) for row in method_rows) / len(method_rows),
                "mean_token_f1": sum(row["quality_metrics"].get("token_f1", 0.0) for row in method_rows) / len(method_rows),
                "mean_compression_ms": sum(row["latency"]["compression_ms"] for row in method_rows) / len(method_rows),
                "mean_prefill_ms": sum(row["latency"]["prefill_ms"] for row in method_rows) / len(method_rows),
                "mean_decode_ms": sum(row["latency"]["decode_ms"] for row in method_rows) / len(method_rows),
            }
        )
    args.summary.write_text(json.dumps({"rows": len(rows), "summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
