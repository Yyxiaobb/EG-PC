"""Real 7B ablation for related/unrelated sequences and graph persistence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.compressors import (
    EvidenceGraphCompressor,
    NoCompressionCompressor,
    QueryAwareMMRCompressor,
    RiskAwareEvidenceGraphCompressor,
)
from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique
from ladc.evidence_graph import EvidenceGraph
from ladc.experiments.runner import run_request
from ladc.inference import LlamaServerInferenceBackend
from ladc.quality import evaluate_real_request
from ladc.sequences import SequenceSpec, build_related_sequence_specs, build_unrelated_sequence_specs
from ladc.tokenizers import HuggingFaceTokenizer


def _load_all_requests(data_root: Path):
    specs = [
        ("hotpotqa", ("hotpotqa", "hotpot_qa"), "validation", load_hotpotqa),
        ("2wikimultihopqa", ("2wikimultihopqa", "2WikiMultihopQA"), "validation", load_2wikimultihopqa),
        ("musique", ("musique", "MuSiQue"), "dev", load_musique),
    ]
    result = {}
    for dataset, names, split, loader in specs:
        path = next((data_root / name for name in names if (data_root / name).exists()), None)
        if path is not None:
            result[dataset] = loader(path, split=split)
    return result


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        request_metadata = row["metadata"]["request_metadata"]
        groups[(str(request_metadata["sequence_kind"]), str(row["method"]), int(request_metadata["turn_index"]))].append(row)
    summaries = []
    for (kind, method, turn_index), group in sorted(groups.items()):
        metrics = [row["quality_metrics"] for row in group]
        latencies = [float(row["latency"]["end_to_end_ms"]) for row in group]
        compressions = [row["metadata"]["compression_metadata"] for row in group]
        summaries.append(
            {
                "sequence_kind": kind,
                "method": method,
                "turn_index": turn_index,
                "n": len(group),
                "mean_input_tokens": sum(float(row["compressed_prompt_tokens"]) for row in group) / len(group),
                "mean_token_reduction_rate": sum(float(row["token_reduction_rate"]) for row in group) / len(group),
                "mean_token_f1": sum(float(metric.get("token_f1", 0.0)) for metric in metrics) / len(metrics),
                "mean_exact_match": sum(float(metric.get("exact_match", 0.0)) for metric in metrics) / len(metrics),
                "mean_supporting_fact_precision": sum(float(metric.get("supporting_fact_precision", 0.0)) for metric in metrics) / len(metrics),
                "mean_supporting_fact_recall": sum(float(metric.get("supporting_fact_recall", 0.0)) for metric in metrics) / len(metrics),
                "mean_evidence_f1": sum(float(metric.get("evidence_f1", 0.0)) for metric in metrics) / len(metrics),
                "mean_answer_context_support": sum(float(metric.get("answer_context_support", 0.0)) for metric in metrics) / len(metrics),
                "mean_compression_ms": sum(float(row["latency"]["compression_ms"]) for row in group) / len(group),
                "mean_prefill_ms": sum(float(row["latency"]["prefill_ms"]) for row in group) / len(group),
                "mean_decode_ms": sum(float(row["latency"]["decode_ms"]) for row in group) / len(group),
                "mean_end_to_end_ms": sum(latencies) / len(latencies),
                "p50_end_to_end_ms": _percentile(latencies, 0.50),
                "p95_end_to_end_ms": _percentile(latencies, 0.95),
                "mean_historical_candidate_count": sum(float(meta.get("historical_candidate_count", 0.0)) for meta in compressions) / len(compressions),
                "mean_reused_evidence_count": sum(float(meta.get("reused_evidence_count", 0.0)) for meta in compressions) / len(compressions),
            }
        )
    return summaries


def _annotated_request(request, spec: SequenceSpec, turn_index: int):
    return type(request)(
        request_id=request.request_id,
        prompt=request.prompt,
        question=request.question,
        context=request.context,
        references=request.references,
        supporting_facts=request.supporting_facts,
        target_output_tokens=request.target_output_tokens,
        metadata={
            **request.metadata,
            "sequence_id": spec.sequence_id,
            "sequence_kind": spec.kind,
            "turn_index": turn_index,
            "adjacent_relation": spec.adjacent_relations[turn_index - 1] if turn_index else {},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument("--server", type=Path, default=Path("artifacts/llama_cpp/runtime/llama-server.exe"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"))
    parser.add_argument("--output", type=Path, default=Path("results/real_7b_sequence_ablation.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/real_7b_sequence_ablation_summary.json"))
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--max-related-per-dataset", type=int, default=2)
    parser.add_argument("--max-unrelated-per-dataset", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9936)
    args = parser.parse_args()

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    requests_by_dataset = _load_all_requests(args.data_root)
    sequences = build_related_sequence_specs(
        requests_by_dataset,
        sequence_length=args.sequence_length,
        max_sequences_per_dataset=args.max_related_per_dataset,
    )
    sequences += build_unrelated_sequence_specs(
        requests_by_dataset,
        sequence_length=args.sequence_length,
        max_sequences_per_dataset=args.max_unrelated_per_dataset,
    )
    if not sequences:
        raise RuntimeError("no sequences found")

    backend = LlamaServerInferenceBackend(
        server_path=args.server,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path="results/llama_server_sequence_ablation.log",
    )
    rows: list[dict[str, object]] = []
    try:
        with backend:
            backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
            for spec in sequences:
                history_compressor = EvidenceGraphCompressor(
                    graph=EvidenceGraph(),
                    name="evidence_graph_history",
                )
                risk_history_compressor = RiskAwareEvidenceGraphCompressor(
                    graph=EvidenceGraph(),
                    name="risk_aware_evidence_graph",
                )
                for turn_index, raw_request in enumerate(spec.requests):
                    request = _annotated_request(raw_request, spec, turn_index)
                    baseline = run_request(
                        request,
                        compressor=NoCompressionCompressor(),
                        backend=backend,
                        tokenizer=tokenizer,
                        quality_fn=evaluate_real_request,
                    )
                    rows.append(baseline.to_dict())
                    baseline_score = baseline.quality_score
                    query_aware = run_request(
                        request,
                        compressor=QueryAwareMMRCompressor(retain_ratio=0.5),
                        backend=backend,
                        tokenizer=tokenizer,
                        baseline_quality_score=baseline_score,
                        quality_fn=evaluate_real_request,
                    )
                    rows.append(query_aware.to_dict())
                    current_only = run_request(
                        request,
                        compressor=EvidenceGraphCompressor(
                            graph=EvidenceGraph(),
                            name="evidence_graph_current_only",
                        ),
                        backend=backend,
                        tokenizer=tokenizer,
                        baseline_quality_score=baseline_score,
                        quality_fn=evaluate_real_request,
                    )
                    rows.append(current_only.to_dict())
                    risk_aware = run_request(
                        request,
                        compressor=risk_history_compressor,
                        backend=backend,
                        tokenizer=tokenizer,
                        baseline_quality_score=baseline_score,
                        quality_fn=evaluate_real_request,
                    )
                    rows.append(risk_aware.to_dict())
                    history = run_request(
                        request,
                        compressor=history_compressor,
                        backend=backend,
                        tokenizer=tokenizer,
                        baseline_quality_score=baseline_score,
                        quality_fn=evaluate_real_request,
                    )
                    rows.append(history.to_dict())
    finally:
        backend.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "sequence_count": len(sequences),
        "sequence_length": args.sequence_length,
        "sequences": [
            {
                "sequence_id": spec.sequence_id,
                "dataset": spec.dataset,
                "kind": spec.kind,
                "request_ids": [request.request_id for request in spec.requests],
                "adjacent_relations": spec.adjacent_relations,
            }
            for spec in sequences
        ],
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "summaries": _summarize(rows),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
