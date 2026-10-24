"""Full MuSiQue dev comparison by native 2/3/4-hop decomposition labels."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ladc.compressors import (  # noqa: E402
    AbstractiveFusedEvidenceGraphCompressor,
    FusedEvidenceGraphCompressor,
    LLMLingua2Compressor,
    LocalAbstractiveSummarizer,
    NoCompressionCompressor,
)
from ladc.datasets import load_musique  # noqa: E402
from ladc.experiments.runner import run_request  # noqa: E402
from ladc.inference import LlamaServerInferenceBackend  # noqa: E402
from ladc.quality import evaluate_real_request  # noqa: E402
from ladc.tokenizers import HuggingFaceTokenizer  # noqa: E402


METHODS = ("no_compression", "llmlingua2", "evidence_graph_fused_v2", "evidence_graph_abstractive_v3")


def hop_count(request) -> int:
    match = re.match(r"(\d+)hop", request.request_id)
    if match:
        return int(match.group(1))
    # The loader already carries the native decomposition's support facts.
    raise ValueError(f"cannot determine native MuSiQue hop count from id: {request.request_id}")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * p)))]


def bootstrap_paired(rows_a, rows_b, field, *, reps=2000, seed=42):
    paired = [(float(a[field]), float(b[field])) for a, b in zip(rows_a, rows_b)]
    if not paired:
        return None
    differences = [a - b for a, b in paired]
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        samples.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    return {
        "mean_difference_a_minus_b": sum(differences) / len(differences),
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "paired_n": len(paired),
    }


def summarize(rows: list[dict], expected_questions: int) -> dict:
    by_hop_method: dict[tuple[str, int], list[dict]] = defaultdict(list)
    by_id_method: dict[tuple[str, str], dict] = {}
    for row in rows:
        hop = int(row["metadata"]["request_metadata"]["hop_count"])
        by_hop_method[(row["method"], hop)].append(row)
        by_id_method[(row["request_id"], row["method"])] = row

    metrics = {}
    for hop in (2, 3, 4):
        for method in METHODS:
            items = by_hop_method[(method, hop)]
            if not items:
                continue
            avg = lambda fn: sum(fn(r) for r in items) / len(items)
            latency = [float(r["latency"]["end_to_end_ms"]) for r in items]
            metrics[f"{hop}_hop/{method}"] = {
                "n": len(items),
                "token_f1": avg(lambda r: float(r["quality_metrics"].get("token_f1", 0))),
                "exact_match": avg(lambda r: float(r["quality_metrics"].get("exact_match", 0))),
                "answer_context_support": avg(lambda r: float(r["quality_metrics"].get("answer_context_support", 0))),
                "supporting_fact_precision": avg(lambda r: float(r["quality_metrics"].get("supporting_fact_precision", 0))),
                "supporting_fact_recall": avg(lambda r: float(r["quality_metrics"].get("supporting_fact_recall", 0))),
                "evidence_f1": avg(lambda r: float(r["quality_metrics"].get("evidence_f1", 0))),
                "mean_original_tokens": avg(lambda r: float(r["original_prompt_tokens"])),
                "mean_compressed_tokens": avg(lambda r: float(r["compressed_prompt_tokens"])),
                "token_reduction_rate": avg(lambda r: float(r["token_reduction_rate"])),
                "compression_wallclock_ms": avg(lambda r: float(r["metadata"].get("compression_wallclock_ms", 0))),
                "prefill_ms": avg(lambda r: float(r["latency"]["prefill_ms"])),
                "decode_ms": avg(lambda r: float(r["latency"]["decode_ms"])),
                "mean_end_to_end_ms": sum(latency) / len(latency),
                "p50_end_to_end_ms": percentile(latency, 0.50),
                "p95_end_to_end_ms": percentile(latency, 0.95),
            }

        reference_method = "no_compression"
        for method in METHODS:
            if method == reference_method:
                continue
            paired_a, paired_b = [], []
            for (rid, candidate), b in by_id_method.items():
                if candidate != method:
                    continue
                a = by_id_method.get((rid, reference_method))
                if a and int(a["metadata"]["request_metadata"]["hop_count"]) == hop:
                    paired_a.append(a["quality_metrics"])
                    paired_b.append(b["quality_metrics"])
            metrics[f"{hop}_hop/paired_no_compression_minus_{method}"] = {
                "token_f1": bootstrap_paired(paired_a, paired_b, "token_f1", seed=hop * 17 + len(method)),
                "exact_match": bootstrap_paired(paired_a, paired_b, "exact_match", seed=hop * 31 + len(method)),
            }

    counts = {str(h): sum(1 for rid, method in by_id_method if method == METHODS[0]
                          and int(by_id_method[(rid, method)]["metadata"]["request_metadata"]["hop_count"]) == h)
              for h in (2, 3, 4)}
    return {
        "complete": all(counts[str(h)] > 0 for h in (2, 3, 4))
                      and len(by_id_method) == expected_questions * len(METHODS),
        "expected_questions": expected_questions,
        "completed_questions": sum(counts.values()),
        "hop_counts": counts,
        "row_count": len(rows),
        "summaries": metrics,
    }


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset\MuSiQue"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument("--server", type=Path, default=Path("artifacts/llama_cpp/runtime/llama-server.exe"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"))
    parser.add_argument("--llmlingua-model", type=Path, default=Path(r"D:\HF_dl\model\llmlingua-2-bert-base-multilingual-cased-meetingbank"))
    parser.add_argument("--abstractive-model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--abstractive-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output-prefix", type=Path, default=Path("results/musique_full_hop_abstractive_comparison"))
    parser.add_argument("--budget-ratio", type=float, default=0.50)
    parser.add_argument("--max-new-tokens", type=int, default=-1,
                        help="llama.cpp prediction limit; -1 generates until EOS or context limit")
    parser.add_argument("--port", type=int, default=9954)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    requests = load_musique(args.data_root, split="dev")
    for request in requests:
        hops = hop_count(request)
        if hops not in (2, 3, 4):
            raise ValueError(f"unexpected MuSiQue hop count {hops} for {request.request_id}")
        request.metadata["hop_count"] = hops
        request.metadata["hop_bucket"] = f"{hops}_hop"
    counts = {str(h): sum(hop_count(r) == h for r in requests) for h in (2, 3, 4)}
    output = args.output_prefix.with_suffix(".jsonl")
    summary_path = args.output_prefix.parent / f"{args.output_prefix.name}_summary.json"
    plan_path = args.output_prefix.parent / f"{args.output_prefix.name}_plan.json"
    plan = {
        "dataset": "MuSiQue",
        "split": "dev",
        "dataset_size": len(requests),
        "native_hop_annotation": "leading Nhop prefix of MuSiQue question id (the dataset's native hop label)",
        "hop_counts": counts,
        "methods": list(METHODS),
        "budget_ratio": args.budget_ratio,
        "model": str(args.model),
        "llmlingua_model": str(args.llmlingua_model),
        "abstractive_model": str(args.abstractive_model),
        "max_new_tokens": args.max_new_tokens,
        "generation_stop": "EOS or 8192-token llama.cpp context limit" if args.max_new_tokens == -1 else "configured n_predict cap or EOS",
        "latency_definition": "compressor invocation through complete generation return; evaluation excluded",
        "protocol": "standalone per-question inference; fresh graph for V2/V3 each request; method order rotates by request; no cross-question history reuse",
        "note": "V3 uses graph-selected source sentences followed by a local Qwen2.5-0.5B abstractive summary; raw source sentence IDs remain provenance anchors. Summary latency is included because this standalone question has no previous-turn idle window.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("resume rejected: experiment plan changed")
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()] if output.exists() else []
    else:
        if output.exists() or plan_path.exists():
            raise FileExistsError(f"experiment files already exist for {args.output_prefix}; use --resume or another prefix")
        rows = []
        atomic_json(plan_path, plan)
    completed = {(r["request_id"], r["method"]) for r in rows}
    if len(completed) != len(rows):
        raise RuntimeError("duplicate rows found in checkpoint")
    print(json.dumps({"plan": plan, "completed_rows": len(rows)}, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    lingua = LLMLingua2Compressor(
        str(args.llmlingua_model), retain_ratio=args.budget_ratio,
        return_segments=True, enforce_budget=True, model_config={"local_files_only": True},
    )
    summarizer = LocalAbstractiveSummarizer(str(args.abstractive_model), device=args.abstractive_device, max_new_tokens=256)
    backend = LlamaServerInferenceBackend(
        server_path=args.server, model_path=args.model, tokenizer_path=args.tokenizer,
        port=args.port, max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path=str(args.output_prefix.parent / f"{args.output_prefix.name}_server.log"),
    )
    methods = {"no_compression": NoCompressionCompressor(), "llmlingua2": lingua}
    started = time.perf_counter()
    try:
        with backend, output.open("a", encoding="utf-8") as handle:
            backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
            lingua.compress_request(requests[0], tokenizer=tokenizer, target_budget_ratio=args.budget_ratio)
            for index, request in enumerate(requests):
                order = list(METHODS[index % len(METHODS):]) + list(METHODS[:index % len(METHODS)])
                for method in order:
                    if (request.request_id, method) in completed:
                        continue
                    if method in methods:
                        compressor = methods[method]
                    elif method == "evidence_graph_fused_v2":
                        compressor = FusedEvidenceGraphCompressor(retain_ratio=args.budget_ratio)
                    else:
                        compressor = AbstractiveFusedEvidenceGraphCompressor(
                            retain_ratio=args.budget_ratio, summarizer=summarizer,
                        )
                    try:
                        row = run_request(
                            request, compressor=compressor, backend=backend, tokenizer=tokenizer,
                            quality_fn=evaluate_real_request,
                            target_budget_ratio=None if method == "no_compression" else args.budget_ratio,
                        ).to_dict()
                        row["metadata"]["compression_metadata"].pop("original_context_segments", None)
                        row["metadata"]["compression_metadata"].pop("compressed_context_segments", None)
                        row["metadata"]["request_metadata"]["hop_count"] = request.metadata["hop_count"]
                        row["metadata"]["request_metadata"]["hop_bucket"] = request.metadata["hop_bucket"]
                    finally:
                        if method in {"evidence_graph_fused_v2", "evidence_graph_abstractive_v3"}:
                            compressor.close()
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    rows.append(row)
                    completed.add((request.request_id, method))
                if (index + 1) % 25 == 0 or index + 1 == len(requests):
                    summary = {**plan, **summarize(rows, len(requests)), "elapsed_s": time.perf_counter() - started}
                    atomic_json(summary_path, summary)
                    print(json.dumps({"questions_processed": index + 1, "rows": len(rows), "elapsed_s": round(time.perf_counter() - started, 1), "hop_counts": counts}, ensure_ascii=False), flush=True)
    finally:
        backend.close()
    summary = {**plan, **summarize(rows, len(requests)), "elapsed_s": time.perf_counter() - started}
    atomic_json(summary_path, summary)
    print(f"COMPLETE: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
