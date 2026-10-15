"""Multi-turn 7B experiment on a per-dataset sample of the QA benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

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
from ladc.sequences import SequenceSpec, _context_entities, _query_entities, relation_features  # noqa: E402
from ladc.tokenizers import HuggingFaceTokenizer  # noqa: E402


def _sample_requests(data_root: Path, fraction: float, seed: int):
    specs = [
        ("hotpotqa", ("hotpotqa", "hotpot_qa"), "validation", load_hotpotqa),
        ("2wikimultihopqa", ("2wikimultihopqa", "2WikiMultihopQA"), "validation", load_2wikimultihopqa),
        ("musique", ("musique", "MuSiQue"), "dev", load_musique),
    ]
    del seed  # sequence sampling uses a separate seeded shuffle below
    sampled: dict[str, list] = {}
    counts: dict[str, dict[str, int | float]] = {}
    for dataset, names, split, loader in specs:
        path = next((data_root / name for name in names if (data_root / name).exists()), None)
        if path is None:
            continue
        requests = loader(path, split=split)
        count = max(1, math.ceil(len(requests) * fraction))
        sampled[dataset] = requests
        counts[dataset] = {
            "available": len(requests),
            "target_turns": count,
            "target_fraction": count / len(requests),
        }
    return sampled, counts


def _annotate(request, spec: SequenceSpec, turn_index: int):
    return replace(
        request,
        metadata={
            **request.metadata,
            "session_id": spec.sequence_id,
            "question_family_id": spec.sequence_id,
            "sequence_id": spec.sequence_id,
            "sequence_kind": spec.kind,
            "turn_index": turn_index,
            "adjacent_relation": spec.adjacent_relations[turn_index - 1] if turn_index else {},
        },
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))]


def _summarize(rows: list[dict[str, object]]):
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        meta = row["metadata"]["request_metadata"]
        groups[(str(row["method"]), int(meta["turn_index"]))].append(row)
    summaries = []
    for (method, turn_index), items in sorted(groups.items()):
        avg = lambda fn: sum(fn(row) for row in items) / len(items)
        latency = [float(row["latency"]["end_to_end_ms"]) for row in items]
        compression = [row["metadata"].get("compression_metadata", {}) for row in items]
        summaries.append({
            "method": method,
            "turn_index": turn_index,
            "n": len(items),
            "session_count": len({row["metadata"]["request_metadata"]["session_id"] for row in items}),
            "mean_input_tokens": avg(lambda row: float(row["compressed_prompt_tokens"])),
            "mean_token_reduction_rate": avg(lambda row: float(row["token_reduction_rate"])),
            "mean_token_f1": avg(lambda row: float(row["quality_metrics"].get("token_f1", 0.0))),
            "mean_exact_match": avg(lambda row: float(row["quality_metrics"].get("exact_match", 0.0))),
            "mean_evidence_f1": avg(lambda row: float(row["quality_metrics"].get("evidence_f1", 0.0))),
            "mean_compression_ms": avg(lambda row: float(row["latency"]["compression_ms"])),
            "mean_prefill_ms": avg(lambda row: float(row["latency"]["prefill_ms"])),
            "mean_decode_ms": avg(lambda row: float(row["latency"]["decode_ms"])),
            "mean_end_to_end_ms": sum(latency) / len(latency),
            "p50_end_to_end_ms": _percentile(latency, 0.5),
            "p95_end_to_end_ms": _percentile(latency, 0.95),
            "mean_reused_evidence_count": avg(lambda row: float(row["metadata"].get("compression_metadata", {}).get("reused_evidence_count", 0))),
            "mean_historical_candidate_count": avg(lambda row: float(row["metadata"].get("compression_metadata", {}).get("historical_candidate_count", 0))),
            "mean_history_compression_ms": avg(lambda row: float(row["metadata"].get("compression_metadata", {}).get("history_compression_ms", 0))),
            "history_completed_before_next_turn_rate": avg(lambda row: float(bool(row["metadata"].get("compression_metadata", {}).get("history_compression_completed_before_next_request", False)))),
        })
    return summaries


def _build_sequences(sampled: dict[str, list], sequence_length: int, max_sequences: int, fraction: float, seed: int):
    # Build related chains with an inverted title index. This keeps planning
    # fast at 10% dataset scale while using only runtime-visible metadata.
    specs: list[SequenceSpec] = []
    rng = random.Random(seed)
    for dataset, source in sampled.items():
        requests = [request for request in source if request.references and request.metadata.get("context_titles")]
        rng.shuffle(requests)
        target_turns = max(1, math.ceil(len(source) * fraction))
        target_sequences = min(max_sequences, target_turns // sequence_length)
        title_sets = {
            request.request_id: {
                str(title).casefold().strip()
                for title in request.metadata.get("context_titles", ())
                if str(title).strip()
            }
            for request in requests
        }
        query_entities = {request.request_id: _query_entities(request) for request in requests}
        context_entities = {request.request_id: _context_entities(request) for request in requests}
        relation_keys = {
            request.request_id: (
                {f"title:{v}" for v in title_sets[request.request_id]}
                | {f"query:{v}" for v in query_entities[request.request_id]}
                | {f"context:{v}" for v in context_entities[request.request_id]}
            )
            for request in requests
        }
        by_key: dict[str, list] = defaultdict(list)
        for request in requests:
            for key in relation_keys[request.request_id]:
                by_key[key].append(request)
        unused = {request.request_id: request for request in requests}
        sequence_number = 0
        shuffled_seeds = list(requests)
        rng.shuffle(shuffled_seeds)
        while unused and sequence_number < target_sequences:
            seed_request = next((request for request in shuffled_seeds if request.request_id in unused), None)
            if seed_request is None:
                break
            del unused[seed_request.request_id]
            chain = [seed_request]
            relations = []
            while len(chain) < sequence_length:
                candidates = {
                    candidate.request_id: candidate
                    for key in relation_keys[chain[-1].request_id]
                    for candidate in by_key.get(key, ())
                    if candidate.request_id in unused
                }
                if not candidates:
                    break
                next_request = min(
                    candidates.values(),
                    key=lambda candidate: (
                        -len(relation_keys[chain[-1].request_id] & relation_keys[candidate.request_id]),
                        rng.random(),
                    ),
                )
                relations.append(relation_features(chain[-1], next_request))
                chain.append(next_request)
                del unused[next_request.request_id]
            if len(chain) == sequence_length:
                specs.append(SequenceSpec(
                    sequence_id=f"{dataset}_related_{sequence_number}",
                    dataset=dataset,
                    kind="related",
                    requests=tuple(chain),
                    adjacent_relations=tuple(relations),
                ))
                sequence_number += 1
            else:
                # Return the tail so it can seed another chain; never split a
                # turn across sessions.
                for request in reversed(chain[1:]):
                    unused[request.request_id] = request
                if not unused:
                    break
                # Avoid looping forever when the remaining pool has no chain.
                if len(chain) == 1:
                    break
    used_by_dataset: dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        used_by_dataset[spec.dataset].update(request.request_id for request in spec.requests)
    coverage = {
        dataset: {
            "target_questions": max(1, math.ceil(len(sampled[dataset]) * fraction)),
            "used_in_multiturn_sessions": len(used_by_dataset.get(dataset, set())),
            "unused_target_questions": max(0, math.ceil(len(sampled[dataset]) * fraction) - len(used_by_dataset.get(dataset, set()))),
        }
        for dataset, requests in sampled.items()
    }
    return specs, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument("--server", type=Path, default=Path("artifacts/llama_cpp/runtime/llama-server.exe"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"))
    parser.add_argument("--llmlingua-model", type=Path, default=Path(r"D:\HF_dl\model\llmlingua-2-bert-base-multilingual-cased-meetingbank"))
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--max-sequences-per-dataset", type=int, default=10000)
    parser.add_argument("--inter-turn-wait-ms", type=float, default=1000.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9944)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/real_7b_multiturn_10pct.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/real_7b_multiturn_10pct_summary.json"))
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    sampled, dataset_counts = _sample_requests(args.data_root, args.fraction, args.seed)
    sequences, coverage = _build_sequences(sampled, args.sequence_length, args.max_sequences_per_dataset, args.fraction, args.seed)
    plan = {
        "fraction_per_dataset": args.fraction,
        "seed": args.seed,
        "dataset_counts": dataset_counts,
        "sequence_count": len(sequences),
        "sequence_length": args.sequence_length,
        "request_count": sum(len(spec.requests) for spec in sequences),
        "coverage": coverage,
        "sequences": [{"sequence_id": s.sequence_id, "dataset": s.dataset, "request_ids": [r.request_id for r in s.requests]} for s in sequences],
    }
    if not sequences:
        raise RuntimeError("no related multi-turn sequences found in the sampled data")
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=True, indent=2))
        return

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    backend = LlamaServerInferenceBackend(
        server_path=args.server,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path="results/llama_server_multiturn_10pct.log",
    )
    session_graph = SessionEvidenceGraphCompressor()
    combo = PostGenerationLLMLingua2MemoryCompressor(
        str(args.llmlingua_model), retain_ratio=0.50, model_config={"local_files_only": True}
    )
    llm = LLMLingua2Compressor(
        str(args.llmlingua_model), retain_ratio=0.50, model_config={"local_files_only": True}
    )
    rows: list[dict[str, object]] = []
    try:
        with backend:
            backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
            for spec in sequences:
                previous_combo_request_id = None
                combo_rows: list[dict[str, object]] = []
                for turn_index, raw_request in enumerate(spec.requests):
                    if previous_combo_request_id is not None:
                        if args.inter_turn_wait_ms > 0:
                            time.sleep(args.inter_turn_wait_ms / 1000.0)
                        combo.mark_completed_before_next_request(previous_combo_request_id)
                        for prev in combo_rows:
                            if prev["request_id"] == previous_combo_request_id:
                                prev["metadata"]["compression_metadata"].update(combo.history_audit(previous_combo_request_id))
                                break
                    request = _annotate(raw_request, spec, turn_index)
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
                        if compressor is combo:
                            row["metadata"]["compression_metadata"].update(combo.history_audit(request.request_id))
                            combo_rows.append(row)
                            previous_combo_request_id = request.request_id
                        rows.append(row)
            combo.wait_for_history()
            for row in rows:
                if row["method"] == combo.name:
                    row["metadata"]["compression_metadata"].update(combo.history_audit(row["request_id"]))
    finally:
        combo.close()
        session_graph.close()
        backend.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        **plan,
        "rows": len(rows),
        "inter_turn_wait_ms": args.inter_turn_wait_ms,
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
