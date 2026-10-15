"""Run the four-method 7B comparison on an equal fraction of each dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.compressors import (  # noqa: E402
    LLMLingua2Compressor,
    NoCompressionCompressor,
    PostGenerationLLMLingua2MemoryCompressor,
    SessionEvidenceGraphCompressor,
)
from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique  # noqa: E402
from ladc.experiments.runner import run_request  # noqa: E402
from ladc.inference import LlamaServerInferenceBackend  # noqa: E402
from ladc.quality import evaluate_real_request  # noqa: E402
from ladc.tokenizers import HuggingFaceTokenizer  # noqa: E402


def _load_fraction(data_root: Path, fraction: float, seed: int):
    specs = [
        ("hotpotqa", ("hotpotqa", "hotpot_qa"), "validation", load_hotpotqa),
        ("2wikimultihopqa", ("2wikimultihopqa", "2WikiMultihopQA"), "validation", load_2wikimultihopqa),
        ("musique", ("musique", "MuSiQue"), "dev", load_musique),
    ]
    rng = random.Random(seed)
    selected = []
    counts = {}
    for dataset, names, split, loader in specs:
        path = next((data_root / name for name in names if (data_root / name).exists()), None)
        if path is None:
            continue
        requests = loader(path, split=split)
        count = max(1, math.ceil(len(requests) * fraction))
        chosen = rng.sample(requests, count)
        counts[dataset] = {"total": len(requests), "sampled": len(chosen), "fraction": count / len(requests)}
        for request in chosen:
            # Each sampled benchmark question is an independent application
            # session, so its graph cannot leak into another sample.
            selected.append((dataset, replace(request, metadata={**request.metadata, "session_id": f"{dataset}:{request.request_id}"})))
    return selected, counts


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))]


def _summarize(rows: list[dict[str, object]]):
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)
    result = []
    for method, items in sorted(groups.items()):
        avg = lambda fn: sum(fn(row) for row in items) / len(items)
        latencies = [float(row["latency"]["end_to_end_ms"]) for row in items]
        compression = [row["metadata"].get("compression_metadata", {}) for row in items]
        result.append({
            "method": method,
            "n": len(items),
            "mean_input_tokens": avg(lambda row: float(row["compressed_prompt_tokens"])),
            "mean_token_reduction_rate": avg(lambda row: float(row["token_reduction_rate"])),
            "mean_token_f1": avg(lambda row: float(row["quality_metrics"].get("token_f1", 0.0))),
            "mean_exact_match": avg(lambda row: float(row["quality_metrics"].get("exact_match", 0.0))),
            "mean_evidence_f1": avg(lambda row: float(row["quality_metrics"].get("evidence_f1", 0.0))),
            "mean_compression_ms": avg(lambda row: float(row["latency"]["compression_ms"])),
            "mean_prefill_ms": avg(lambda row: float(row["latency"]["prefill_ms"])),
            "mean_decode_ms": avg(lambda row: float(row["latency"]["decode_ms"])),
            "mean_end_to_end_ms": sum(latencies) / len(latencies),
            "p50_end_to_end_ms": _percentile(latencies, 0.50),
            "p95_end_to_end_ms": _percentile(latencies, 0.95),
            "mean_history_compression_ms": avg(lambda row: float(row["metadata"].get("compression_metadata", {}).get("history_compression_ms", 0.0))),
            "history_pending_rate_at_record": avg(lambda row: float(bool(row["metadata"].get("compression_metadata", {}).get("history_compression_pending", False)))),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument("--server", type=Path, default=Path("artifacts/llama_cpp/runtime/llama-server.exe"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"))
    parser.add_argument("--llmlingua-model", type=Path, default=Path(r"D:\HF_dl\model\llmlingua-2-bert-base-multilingual-cased-meetingbank"))
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9943)
    parser.add_argument("--output", type=Path, default=Path("results/real_7b_post_generation_fraction_1pct.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/real_7b_post_generation_fraction_1pct_summary.json"))
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    requests, dataset_counts = _load_fraction(args.data_root, args.fraction, args.seed)
    if not requests:
        raise RuntimeError("no evaluable requests found")

    backend = LlamaServerInferenceBackend(
        server_path=args.server,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path="results/llama_server_post_generation_fraction.log",
    )
    session_graph = SessionEvidenceGraphCompressor()
    combo = PostGenerationLLMLingua2MemoryCompressor(
        str(args.llmlingua_model),
        retain_ratio=0.50,
        model_config={"local_files_only": True},
    )
    llm = LLMLingua2Compressor(
        str(args.llmlingua_model),
        retain_ratio=0.50,
        model_config={"local_files_only": True},
    )
    rows: list[dict[str, object]] = []
    try:
        with backend:
            backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
            for dataset, request in requests:
                baseline = run_request(request, compressor=NoCompressionCompressor(), backend=backend, tokenizer=tokenizer, quality_fn=evaluate_real_request)
                rows.append(baseline.to_dict())
                for compressor in (session_graph, llm, combo):
                    result = run_request(
                        request,
                        compressor=compressor,
                        backend=backend,
                        tokenizer=tokenizer,
                        baseline_quality_score=baseline.quality_score,
                        quality_fn=evaluate_real_request,
                    )
                    row = result.to_dict()
                    row["metadata"]["request_metadata"]["sample_dataset"] = dataset
                    if compressor is combo:
                        row["metadata"]["compression_metadata"].update(combo.history_audit(request.request_id))
                    rows.append(row)
    finally:
        combo.wait_for_history()
        # Independent sessions have no next-turn reuse, but all asynchronous
        # history measurements are finalized before serializing the rows.
        for row in rows:
            if row["method"] == combo.name:
                row["metadata"]["compression_metadata"].update(combo.history_audit(row["request_id"]))
        combo.close()
        session_graph.close()
        backend.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "rows": len(rows),
        "fraction_per_dataset": args.fraction,
        "seed": args.seed,
        "dataset_counts": dataset_counts,
        "methods": ["no_compression", "evidence_graph_dual_path", "llmlingua2", "evidence_graph_post_llmlingua2"],
        "model": str(args.model),
        "llmlingua_model": str(args.llmlingua_model),
        "summaries": _summarize(rows),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
