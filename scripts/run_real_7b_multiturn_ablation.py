"""Controlled 20% multi-turn ablation for V2 evidence retrieval and packing."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import time
import os
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ladc.compressors import FusedEvidenceGraphCompressor, QueryAwareMMRCompressor
from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique
from ladc.experiments.runner import run_request
from ladc.inference import LlamaServerInferenceBackend
from ladc.quality import evaluate_real_request
from ladc.tokenizers import HuggingFaceTokenizer
from scripts.run_real_7b_multiturn_fraction import _annotate, _build_sequences, _sample_requests


METHODS = [
    "evidence_graph_fused_v2",
    "fused_v2_no_history_budget",
    "fused_v2_current_only",
    "query_aware_bm25_mmr",
]


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        meta = row["metadata"]["request_metadata"]
        for key in (("overall", "all", row["method"]),
                    ("dataset", meta["dataset"], row["method"]),
                    ("turn", str(int(meta["turn_index"]) + 1), row["method"])):
            groups.setdefault(key, []).append(row)
    summaries = []
    for (group, value, method), items in sorted(groups.items()):
        avg = lambda fn: sum(fn(row) for row in items) / len(items)
        latency = [row["latency"]["end_to_end_ms"] for row in items]
        cms = [row["metadata"].get("compression_metadata", {}) for row in items]
        summaries.append({
            "group": group, "value": value, "method": method, "n": len(items),
            "token_f1": avg(lambda r: r["quality_metrics"]["token_f1"]),
            "exact_match": avg(lambda r: r["quality_metrics"]["exact_match"]),
            "supporting_fact_precision": avg(lambda r: r["quality_metrics"].get("supporting_fact_precision", 0.0)),
            "supporting_fact_recall": avg(lambda r: r["quality_metrics"].get("supporting_fact_recall", 0.0)),
            "evidence_f1": avg(lambda r: r["quality_metrics"].get("evidence_f1", 0.0)),
            "input_tokens": avg(lambda r: r["compressed_prompt_tokens"]),
            "token_reduction_rate": avg(lambda r: r["token_reduction_rate"]),
            "compression_wallclock_ms": avg(lambda r: r["metadata"]["compression_wallclock_ms"]),
            "prefill_ms": avg(lambda r: r["latency"]["prefill_ms"]),
            "decode_ms": avg(lambda r: r["latency"]["decode_ms"]),
            "end_to_end_wallclock_ms": sum(latency) / len(latency),
            "p50_end_to_end_ms": sorted(latency)[round((len(latency) - 1) * 0.5)],
            "p95_end_to_end_ms": sorted(latency)[round((len(latency) - 1) * 0.95)],
            "reused_evidence_count": avg(lambda r: r["metadata"]["compression_metadata"].get("reused_evidence_count", 0)),
            "historical_candidate_count": avg(lambda r: r["metadata"]["compression_metadata"].get("historical_candidate_count", 0)),
            "selected_evidence_count": avg(lambda r: r["metadata"]["compression_metadata"].get("selected_evidence_count", 0)),
            "retrieval_winners": dict(Counter(c.get("retrieval_audit", {}).get("winner", "n/a") for c in cms)),
        })
    return summaries


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument("--server", type=Path, default=Path("artifacts/llama_cpp/runtime/llama-server.exe"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"))
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-ratio", type=float, default=0.50)
    parser.add_argument("--inter-turn-wait-ms", type=float, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9957)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=Path("results/real_7b_multiturn_v2_20pct_ablation"))
    args = parser.parse_args()

    sampled, counts = _sample_requests(args.data_root, args.fraction, args.seed)
    specs, coverage = _build_sequences(sampled, 3, 10000, args.fraction, args.seed)
    random.Random(args.seed + 1).shuffle(specs)
    previous_plan_path = Path("results/real_7b_multiturn_v2_20pct_three_methods_plan.json")
    previous_plan = json.loads(previous_plan_path.read_text(encoding="utf-8"))
    previous_sequences = [(s["sequence_id"], s["request_ids"]) for s in previous_plan["sequences"]]
    actual_sequences = [(s.sequence_id, [r.request_id for r in s.requests]) for s in specs]
    if actual_sequences != previous_sequences:
        raise RuntimeError("sampled sessions differ from the completed 20% experiment")

    code_files = [Path(__file__), Path("scripts/run_real_7b_multiturn_v2.py"),
                  Path("scripts/run_real_7b_multiturn_fraction.py"),
                  Path("src/ladc/compressors/fused_graph.py"),
                  Path("src/ladc/compressors/query_aware.py"),
                  Path("src/ladc/evidence_graph.py"), Path("src/ladc/evidence_builder.py"),
                  Path("src/ladc/quality.py"), Path("src/ladc/experiments/runner.py")]
    plan = {
        "version": 1, "fraction_per_dataset": args.fraction, "seed": args.seed,
        "dataset_counts": counts, "coverage": coverage, "sequence_length": 3,
        "sequence_count": len(specs), "request_count": sum(len(s.requests) for s in specs),
        "methods": METHODS, "expected_generation_count": sum(len(s.requests) for s in specs) * len(METHODS),
        "budget_ratio": args.budget_ratio,
        "budget_policy": "same maximum native prompt token budget per request; actual usage reported",
        "inter_turn_wait_ms": args.inter_turn_wait_ms,
        "sequence_sampling": "exact request IDs and order from the completed 20% three-method experiment",
        "model": str(args.model), "max_new_tokens": args.max_new_tokens,
        "ablation_definitions": {
            "evidence_graph_fused_v2": "Persistent per-session V2, historical evidence available and capped at 20% of context budget.",
            "fused_v2_no_history_budget": "Persistent per-session V2 graph; historical evidence candidate budget set to zero.",
            "fused_v2_current_only": "Fresh V2 compressor/graph for every request; isolates current-turn graph selection without history.",
            "query_aware_bm25_mmr": "Fresh BM25/MMR extractive compressor for every request; no graph and no history.",
        },
        "previous_experiment_prefix": "results/real_7b_multiturn_v2_20pct_three_methods",
        "previous_plan_sha256": sha256(previous_plan_path.read_bytes()).hexdigest(),
        "code_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in code_files},
        "sequences": [{"sequence_id": s.sequence_id, "dataset": s.dataset, "kind": s.kind,
                       "request_ids": [r.request_id for r in s.requests]} for s in specs],
    }
    prefix = args.output_prefix
    output = prefix.with_suffix(".jsonl")
    summary_path = prefix.parent / (prefix.name + "_summary.json")
    plan_path = prefix.parent / (prefix.name + "_plan.json")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not plan_path.exists() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("resume rejected: protocol, input sequence, or code differs")
    else:
        if output.exists() or summary_path.exists() or plan_path.exists():
            raise FileExistsError(f"outputs already exist for {prefix}")
        atomic_json(plan_path, plan)
    print(json.dumps({"coverage": coverage, "sessions": len(specs), "questions": plan["request_count"],
                      "generations": plan["expected_generation_count"], "methods": METHODS}, ensure_ascii=False), flush=True)
    if args.plan_only:
        return

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    backend = LlamaServerInferenceBackend(server_path=args.server, model_path=args.model,
        tokenizer_path=args.tokenizer, port=args.port, max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path=str(prefix.parent / (prefix.name + "_server.log")))
    rows: list[dict] = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] if args.resume and output.exists() else []
    completed: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        session_id = row["metadata"]["request_metadata"]["session_id"]
        completed[(session_id, row["method"])] += 1
    if any(count != 3 for count in completed.values()):
        raise RuntimeError("checkpoint contains an incomplete session/method block")
    started = time.perf_counter()
    with backend, output.open("a", encoding="utf-8") as handle:
        backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
        for session_index, spec in enumerate(specs):
            order = METHODS[session_index % len(METHODS):] + METHODS[:session_index % len(METHODS)]
            session_changed = False
            for method in order:
                if completed[(spec.sequence_id, method)] == 3:
                    continue
                if method in ("evidence_graph_fused_v2", "fused_v2_no_history_budget"):
                    compressor = FusedEvidenceGraphCompressor(
                        max_history_fraction=0.20 if method == "evidence_graph_fused_v2" else 0.0)
                    persistent = True
                else:
                    compressor = None
                    persistent = False
                block = []
                try:
                    for turn_index, raw in enumerate(spec.requests):
                        if turn_index:
                            time.sleep(args.inter_turn_wait_ms / 1000)
                        request = _annotate(raw, spec, turn_index)
                        active = compressor
                        if method == "fused_v2_current_only":
                            active = FusedEvidenceGraphCompressor(max_history_fraction=0.0)
                        elif method == "query_aware_bm25_mmr":
                            active = QueryAwareMMRCompressor(retain_ratio=args.budget_ratio)
                        row = run_request(request, compressor=active, backend=backend,
                            tokenizer=tokenizer, quality_fn=evaluate_real_request,
                            target_budget_ratio=args.budget_ratio).to_dict()
                        row["method"] = method
                        row["latency"]["component_sum_ms"] = row["latency"]["end_to_end_ms"]
                        row["latency"]["end_to_end_ms"] = row["metadata"]["request_wallclock_ms"]
                        row["metadata"]["compression_metadata"].pop("original_context_segments", None)
                        row["metadata"]["compression_metadata"].pop("compressed_context_segments", None)
                        block.append(row)
                        if not persistent and hasattr(active, "close"):
                            active.close()
                    if persistent and hasattr(compressor, "wait_for_retrieval"):
                        compressor.wait_for_retrieval()
                finally:
                    if persistent and hasattr(compressor, "close"):
                        compressor.close()
                handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in block))
                handle.flush()
                rows.extend(block)
                completed[(spec.sequence_id, method)] = 3
                session_changed = True
            if session_changed:
                atomic_json(summary_path, {**plan, "complete": len(rows) == plan["expected_generation_count"],
                    "rows": len(rows), "elapsed_s": time.perf_counter() - started, "summaries": summarize(rows)})
            print(json.dumps({"completed_sessions": session_index + 1, "total_sessions": len(specs),
                "rows": len(rows), "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    print(f"COMPLETE: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
