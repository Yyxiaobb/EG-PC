"""Paired multi-turn experiment with independent sessions and wall-clock timing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ladc.compressors import (FusedEvidenceGraphCompressor, LLMLingua2Compressor,
    AsyncAbstractiveEvidenceMemoryCompressor, LocalAbstractiveSummarizer,
    NoCompressionCompressor, PostGenerationLLMLingua2MemoryCompressor, SessionEvidenceGraphCompressor,
    QualityConstrainedGraphRateDistortionCompressor)
from ladc.compressors import (RelationAwareQCGRDCCompressor,
    TripleAwareQCGRDCCompressor, ChainAwareQCGRDCCompressor)
from ladc.experiments.runner import run_request
from ladc.inference import LlamaServerInferenceBackend
from ladc.quality import evaluate_real_request
from ladc.tokenizers import HuggingFaceTokenizer
from scripts.run_real_7b_multiturn_fraction import _annotate, _build_sequences, _sample_requests


ALL_METHODS = ["no_compression", "evidence_graph_dual_path_v1", "llmlingua2",
               "evidence_graph_fused_v2", "evidence_graph_fused_post_llmlingua2_v2",
               "evidence_graph_abstractive_async_v3", "evidence_graph_qc_grdc_v4",
               "evidence_graph_qc_grdc_relation_v1", "evidence_graph_qc_grdc_triple_v1",
               "evidence_graph_qc_grdc_chain_v1"]


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)] if values else None


def summarize(rows):
    def aggregate(items):
        avg = lambda fn: sum(fn(r) for r in items) / len(items)
        cms = [r["metadata"]["compression_metadata"] for r in items]
        background = [c for c in cms if "history_total_ms" in c]
        eligible = [c for c in cms if c.get("has_next_request")]
        latency = [r["latency"]["end_to_end_ms"] for r in items]
        return {"n": len(items),
            "token_f1": avg(lambda r: r["quality_metrics"]["token_f1"]),
            "exact_match": avg(lambda r: r["quality_metrics"]["exact_match"]),
            "evidence_f1_extractive_only": avg(lambda r: r["quality_metrics"]["evidence_f1"]),
            "input_tokens": avg(lambda r: r["compressed_prompt_tokens"]),
            "token_reduction_rate": avg(lambda r: r["token_reduction_rate"]),
            "compression_wallclock_ms": avg(lambda r: r["metadata"]["compression_wallclock_ms"]),
            "prefill_ms": avg(lambda r: r["latency"]["prefill_ms"]),
            "decode_ms": avg(lambda r: r["latency"]["decode_ms"]),
            "end_to_end_wallclock_ms": sum(latency) / len(latency),
            "p50_end_to_end_ms": percentile(latency, 0.5), "p95_end_to_end_ms": percentile(latency, 0.95),
            "component_sum_ms": avg(lambda r: r["latency"]["component_sum_ms"]),
            "reused_evidence_count": sum(c.get("reused_evidence_count", 0) for c in cms) / len(items),
            "compressed_memory_selected_count": sum(c.get("compressed_memory_selected_count", 0) for c in cms),
            "retrieval_winners": dict(Counter(c.get("retrieval_audit", {}).get("winner", "n/a") for c in cms)),
            "history_compute_ms": sum(c["history_compression_ms"] for c in background) / len(background) if background else None,
            "history_queue_ms": sum(c["history_queue_ms"] for c in background) / len(background) if background else None,
            "history_total_ms": sum(c["history_total_ms"] for c in background) / len(background) if background else None,
            "history_supplemented_source_count": sum(c.get("history_supplemented_source_count", 0) for c in background),
            "history_ready_numerator": sum(bool(c.get("history_compression_completed_before_next_request")) for c in eligible),
            "history_ready_denominator": len(eligible),
            "history_ready_rate": sum(bool(c.get("history_compression_completed_before_next_request")) for c in eligible) / len(eligible) if eligible else None,
            "history_errors": [c["history_compression_error"] for c in cms if c.get("history_compression_error")],
            "history_summary_fallback_count": sum(bool(c.get("history_summary_fallback_to_raw")) for c in cms),
            "history_mean_reduction_rate": avg(lambda r: float(r["metadata"]["compression_metadata"].get("history_reduction_rate", 0.0))),
        }
    groups = defaultdict(list)
    for r in rows:
        meta = r["metadata"]["request_metadata"]
        groups[("overall", "all", r["method"])].append(r)
        groups[("dataset", meta["dataset"], r["method"])].append(r)
        groups[("turn", str(int(meta["turn_index"]) + 1), r["method"])].append(r)
    return [{"group": k[0], "value": k[1], "method": k[2], **aggregate(v)}
            for k, v in sorted(groups.items())]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument("--server", type=Path, default=Path("artifacts/llama_cpp/runtime/llama-server.exe"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"))
    parser.add_argument("--llmlingua-model", type=Path, default=Path(r"D:\HF_dl\model\llmlingua-2-bert-base-multilingual-cased-meetingbank"))
    parser.add_argument("--abstractive-model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--abstractive-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--abstractive-max-new-tokens", type=int, default=64)
    parser.add_argument("--fraction", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences-per-dataset", type=int, default=10000)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS,
                        help="methods to compare; default runs all registered methods")
    parser.add_argument("--datasets", nargs="+", choices=("hotpotqa", "2wikimultihopqa", "musique"),
                        default=("hotpotqa", "2wikimultihopqa", "musique"),
                        help="optional dataset subset; defaults to all three datasets")
    parser.add_argument("--budget-ratio", type=float, default=0.50)
    parser.add_argument("--inter-turn-wait-ms", type=float, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9953)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=Path("results/real_7b_multiturn_v2_3pct"))
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or not 0 < args.budget_ratio <= 1 or args.inter_turn_wait_ms < 0:
        parser.error("invalid fraction, budget, or inter-turn wait")
    if len(set(args.methods)) != len(args.methods) or not args.methods:
        parser.error("methods must be a non-empty list without duplicates")
    methods = list(args.methods)
    sampled, dataset_counts = _sample_requests(args.data_root, args.fraction, args.seed)
    specs, coverage = _build_sequences(sampled, 3, args.max_sequences_per_dataset, args.fraction, args.seed)
    random.Random(args.seed + 1).shuffle(specs)
    missing_datasets = set(args.datasets) - set(sampled)
    if missing_datasets:
        raise RuntimeError(f"requested datasets unavailable: {sorted(missing_datasets)}")
    # Filter only after building and shuffling with all datasets so a dataset
    # subset reuses the exact same sequence sample as the full paired run.
    specs = [spec for spec in specs if spec.dataset in args.datasets]
    coverage = {key: value for key, value in coverage.items() if key in args.datasets}
    dataset_counts = {key: value for key, value in dataset_counts.items() if key in args.datasets}
    code_files = sorted([*Path("src/ladc").rglob("*.py"), Path(__file__),
                         Path("scripts/run_real_7b_multiturn_fraction.py")])
    plan = {"version": 3, "fraction_per_dataset": args.fraction, "seed": args.seed,
        "dataset_counts": dataset_counts, "coverage": coverage, "sequence_length": 3,
        "sequence_count": len(specs), "request_count": sum(len(s.requests) for s in specs),
        "methods": methods, "expected_generation_count": sum(len(s.requests) for s in specs) * len(methods),
        "budget_ratio": args.budget_ratio, "budget_policy": "same maximum native prompt token budget; actual usage reported",
        "sequence_sampling": "three-turn chains selected from full split using visible titles/entities; deterministic mixed fallback fills requested fraction when linked turns run out; not natural dialogue",
        "inter_turn_wait_ms": args.inter_turn_wait_ms,
        "method_schedule": "rotate method order across sessions; finish background work before changing methods",
        "latency_definition": "client compression start to complete generation return, excluding evaluation and post-generation work",
        "history_readiness_definition": "measured immediately before actual next combo request; final turns excluded",
        "raw_source_anchors_retained": True, "max_new_tokens": args.max_new_tokens,
        "model": str(args.model), "llmlingua_model": str(args.llmlingua_model),
        "abstractive_model": str(args.abstractive_model), "abstractive_device": args.abstractive_device,
        "abstractive_max_new_tokens": args.abstractive_max_new_tokens,
        "v3_protocol": "turn 1 sends the original prompt unchanged; after each answer, an asynchronous worker summarizes only that turn's raw source evidence and stores a source-only memory node with provenance IDs; later turns fuse current raw evidence with completed session summaries; a pending summary never blocks a request",
        "qc_grdc_protocol": "quality-constrained graph rate-distortion coding; protect query variables, numeric/negation features, graph bridge paths and provenance anchors, then select remaining evidence by marginal observable quality gain per native token cost",
        "datasets": list(args.datasets),
        "code_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in code_files},
        "sequences": [{"sequence_id": s.sequence_id, "dataset": s.dataset, "kind": s.kind,
                       "request_ids": [r.request_id for r in s.requests]} for s in specs]}
    output = args.output_prefix.with_suffix(".jsonl")
    summary_path = args.output_prefix.parent / (args.output_prefix.name + "_summary.json")
    plan_path = args.output_prefix.parent / (args.output_prefix.name + "_plan.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        previous_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous_plan != plan:
            raise RuntimeError("resume rejected: protocol, inputs or code differ")
    else:
        if output.exists() or summary_path.exists():
            raise FileExistsError("output exists; use a different prefix or --resume")
        atomic_json(plan_path, plan)
    print(json.dumps({k: plan[k] for k in ("coverage", "sequence_count", "request_count", "expected_generation_count")}), flush=True)
    if args.plan_only:
        return
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if args.resume and output.exists() else []
    completed = Counter((r["metadata"]["request_metadata"]["session_id"], r["method"]) for r in rows)
    if any(n != 3 for n in completed.values()):
        raise RuntimeError("checkpoint has incomplete method blocks")
    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    llm = LLMLingua2Compressor(str(args.llmlingua_model), retain_ratio=args.budget_ratio,
        return_segments=True, enforce_budget=True, model_config={"local_files_only": True})
    abstractive_summarizer = (LocalAbstractiveSummarizer(str(args.abstractive_model),
        device=args.abstractive_device, max_new_tokens=args.abstractive_max_new_tokens)
        if "evidence_graph_abstractive_async_v3" in methods else None)
    backend = LlamaServerInferenceBackend(server_path=args.server, model_path=args.model,
        tokenizer_path=args.tokenizer, port=args.port, max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path=str(args.output_prefix.parent / (args.output_prefix.name + "_server.log")))
    experiment_started = time.perf_counter()
    with backend, output.open("a", encoding="utf-8") as handle:
        backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
        warm = _annotate(specs[0].requests[0], specs[0], 0)
        llm.compress_request(warm, tokenizer=tokenizer, target_budget_ratio=args.budget_ratio)
        for session_index, spec in enumerate(specs):
            order = methods[session_index % len(methods):] + methods[:session_index % len(methods)]
            for method in order:
                if completed[(spec.sequence_id, method)] == 3:
                    continue
                if method == "no_compression":
                    compressor = NoCompressionCompressor()
                elif method == "llmlingua2":
                    compressor = llm
                elif method == "evidence_graph_dual_path_v1":
                    compressor = SessionEvidenceGraphCompressor(compressor_kwargs={"name": method})
                elif method == "evidence_graph_abstractive_async_v3":
                    compressor = AsyncAbstractiveEvidenceMemoryCompressor(
                        abstractive_summarizer, retain_ratio=args.budget_ratio)
                elif method == "evidence_graph_qc_grdc_v4":
                    compressor = SessionEvidenceGraphCompressor(
                        compressor_class=QualityConstrainedGraphRateDistortionCompressor,
                        compressor_kwargs={"name": method, "quality_floor": 0.92},
                    )
                elif method == "evidence_graph_qc_grdc_relation_v1":
                    compressor = SessionEvidenceGraphCompressor(
                        compressor_class=RelationAwareQCGRDCCompressor,
                        compressor_kwargs={"name": method, "quality_floor": 0.92},
                    )
                elif method == "evidence_graph_qc_grdc_triple_v1":
                    compressor = SessionEvidenceGraphCompressor(
                        compressor_class=TripleAwareQCGRDCCompressor,
                        compressor_kwargs={"name": method, "quality_floor": 0.92},
                    )
                elif method == "evidence_graph_qc_grdc_chain_v1":
                    compressor = SessionEvidenceGraphCompressor(
                        compressor_class=ChainAwareQCGRDCCompressor,
                        compressor_kwargs={"name": method, "quality_floor": 0.92},
                    )
                else:
                    session = SessionEvidenceGraphCompressor(compressor_class=FusedEvidenceGraphCompressor)
                    compressor = (PostGenerationLLMLingua2MemoryCompressor("unused", name=method,
                        session_compressor=session, memory_compressor=llm, reconcile_sources=True)
                        if method.endswith("llmlingua2_v2") else session)
                block = []
                try:
                    for turn_index, raw in enumerate(spec.requests):
                        if turn_index:
                            time.sleep(args.inter_turn_wait_ms / 1000)
                        request = _annotate(raw, spec, turn_index)
                        row = run_request(request, compressor=compressor, backend=backend,
                            tokenizer=tokenizer, quality_fn=evaluate_real_request,
                            target_budget_ratio=None if method == "no_compression" else args.budget_ratio).to_dict()
                        row["method"] = method
                        row["latency"]["component_sum_ms"] = row["latency"]["end_to_end_ms"]
                        row["latency"]["end_to_end_ms"] = row["metadata"]["request_wallclock_ms"]
                        row["metadata"]["compression_metadata"].pop("original_context_segments", None)
                        row["metadata"]["compression_metadata"].pop("compressed_context_segments", None)
                        block.append(row)
                    if hasattr(compressor, "wait_for_history"):
                        compressor.wait_for_history()
                    if hasattr(compressor, "history_audit"):
                        for row in block:
                            audit = compressor.history_audit(row["request_id"])
                            row["metadata"]["compression_metadata"].update(audit)
                            if audit.get("history_compression_error"):
                                raise RuntimeError(audit["history_compression_error"])
                finally:
                    if hasattr(compressor, "close"):
                        compressor.close()
                handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in block))
                handle.flush()
                rows.extend(block)
                completed[(spec.sequence_id, method)] = 3
            atomic_json(summary_path, {**plan, "complete": len(rows) == plan["expected_generation_count"],
                "rows": len(rows), "elapsed_this_run_s": time.perf_counter() - experiment_started,
                "summaries": summarize(rows)})
            print(json.dumps({"completed_sessions": session_index + 1, "total_sessions": len(specs),
                "rows": len(rows), "elapsed_s": round(time.perf_counter() - experiment_started, 1)}), flush=True)
    print(f"COMPLETE: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
