"""Run a real 7B experiment on related multi-turn question sequences."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.compressors import EvidenceGraphCompressor, NoCompressionCompressor, QueryAwareMMRCompressor
from ladc.datasets import load_2wikimultihopqa, load_hotpotqa, load_musique
from ladc.evidence_graph import EvidenceGraph
from ladc.experiments.runner import run_request
from ladc.inference import LlamaServerInferenceBackend
from ladc.quality import evaluate_real_request
from ladc.tokenizers import HuggingFaceTokenizer


def _load_all_requests(data_root: Path):
    specs = [
        ("hotpotqa", ("hotpotqa", "hotpot_qa"), "validation", load_hotpotqa),
        ("2wikimultihopqa", ("2wikimultihopqa", "2WikiMultihopQA"), "validation", load_2wikimultihopqa),
        ("musique", ("musique", "MuSiQue"), "dev", load_musique),
    ]
    result: dict[str, list] = {}
    for dataset, names, split, loader in specs:
        path = next((data_root / name for name in names if (data_root / name).exists()), None)
        if path is not None:
            result[dataset] = loader(path, split=split)
    return result


def _title_set(request) -> set[str]:
    return {
        str(title).casefold().strip()
        for title in request.metadata.get("context_titles", ())
        if str(title).strip()
    }


def build_related_sequences(
    requests_by_dataset: dict[str, list],
    *,
    sequence_length: int,
    max_sequences_per_dataset: int,
) -> list[tuple[str, list]]:
    """Greedily form sequences linked by shared context titles.

    This grouping uses only runtime-visible context titles and question order,
    never answers or supporting-fact annotations.
    """

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    sequences: list[tuple[str, list]] = []
    for dataset, requests in requests_by_dataset.items():
        unused = [request for request in requests if request.references and _title_set(request)]
        unused.sort(key=lambda request: request.request_id)
        for sequence_number in range(max_sequences_per_dataset):
            if not unused:
                break
            seed = unused.pop(0)
            sequence = [seed]
            shared_titles: set[str] = _title_set(seed)
            while len(sequence) < sequence_length:
                candidates = [
                    request
                    for request in unused
                    if _title_set(request) & shared_titles
                ]
                if not candidates:
                    break
                candidates.sort(
                    key=lambda request: (
                        -len(_title_set(request) & shared_titles),
                        request.request_id,
                    )
                )
                next_request = candidates[0]
                unused.remove(next_request)
                sequence.append(next_request)
                shared_titles |= _title_set(next_request)
            if len(sequence) == sequence_length:
                sequences.append((f"{dataset}_sequence_{sequence_number}", sequence))
            else:
                unused[0:0] = sequence[1:]
    return sequences


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    groups: dict[tuple[str, int | str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata", {})
        request_metadata = metadata.get("request_metadata", {})
        sequence_id = request_metadata.get("sequence_id", "unknown")
        turn_index = request_metadata.get("turn_index", "unknown")
        groups[(str(row["method"]), turn_index)].append(row)
    for (method, turn_index), group in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        latency = [float(row["latency"]["end_to_end_ms"]) for row in group]
        metrics = [row.get("quality_metrics", {}) for row in group]
        compression = [row.get("metadata", {}).get("compression_metadata", {}) for row in group]
        summaries.append(
            {
                "method": method,
                "turn_index": turn_index,
                "sequence_count": len({row["metadata"]["request_metadata"].get("sequence_id") for row in group}),
                "n": len(group),
                "mean_input_tokens": sum(float(row["compressed_prompt_tokens"]) for row in group) / len(group),
                "mean_token_reduction_rate": sum(float(row["token_reduction_rate"]) for row in group) / len(group),
                "mean_token_f1": sum(float(metric.get("token_f1", 0.0)) for metric in metrics) / len(metrics),
                "mean_exact_match": sum(float(metric.get("exact_match", 0.0)) for metric in metrics) / len(metrics),
                "mean_answer_context_support": sum(float(metric.get("answer_context_support", 0.0)) for metric in metrics) / len(metrics),
                "mean_supporting_fact_precision": sum(float(metric.get("supporting_fact_precision", 0.0)) for metric in metrics) / len(metrics),
                "mean_supporting_fact_recall": sum(float(metric.get("supporting_fact_recall", 0.0)) for metric in metrics) / len(metrics),
                "mean_evidence_f1": sum(float(metric.get("evidence_f1", 0.0)) for metric in metrics) / len(metrics),
                "mean_end_to_end_ms": sum(latency) / len(latency),
                "p50_end_to_end_ms": _percentile(latency, 0.50),
                "p95_end_to_end_ms": _percentile(latency, 0.95),
                "mean_compression_ms": sum(float(row["latency"]["compression_ms"]) for row in group) / len(group),
                "mean_prefill_ms": sum(float(row["latency"]["prefill_ms"]) for row in group) / len(group),
                "mean_decode_ms": sum(float(row["latency"]["decode_ms"]) for row in group) / len(group),
                "mean_reused_evidence_count": sum(float(meta.get("reused_evidence_count", 0.0)) for meta in compression) / len(compression),
                "mean_historical_candidate_count": sum(float(meta.get("historical_candidate_count", 0.0)) for meta in compression) / len(compression),
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
    parser.add_argument("--output", type=Path, default=Path("results/real_7b_related_sequences.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/real_7b_related_sequences_summary.json"))
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--max-sequences-per-dataset", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9933)
    args = parser.parse_args()

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    requests_by_dataset = _load_all_requests(args.data_root)
    sequences = build_related_sequences(
        requests_by_dataset,
        sequence_length=args.sequence_length,
        max_sequences_per_dataset=args.max_sequences_per_dataset,
    )
    if not sequences:
        raise RuntimeError("no related question sequences found")

    backend = LlamaServerInferenceBackend(
        server_path=args.server,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path="results/llama_server_related_sequences.log",
    )
    rows: list[dict[str, object]] = []
    try:
        with backend:
            backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
            for sequence_id, sequence in sequences:
                graph_compressor = EvidenceGraphCompressor(
                    graph=EvidenceGraph(),
                    name="evidence_graph_dynamic",
                )
                for turn_index, original_request in enumerate(sequence):
                    request = type(original_request)(
                        request_id=original_request.request_id,
                        prompt=original_request.prompt,
                        question=original_request.question,
                        context=original_request.context,
                        references=original_request.references,
                        supporting_facts=original_request.supporting_facts,
                        target_output_tokens=original_request.target_output_tokens,
                        metadata={
                            **original_request.metadata,
                            "sequence_id": sequence_id,
                            "turn_index": turn_index,
                            "sequence_shared_titles": sorted(_title_set(original_request)),
                        },
                    )
                    baseline = run_request(
                        request,
                        compressor=NoCompressionCompressor(),
                        backend=backend,
                        tokenizer=tokenizer,
                        quality_fn=evaluate_real_request,
                    )
                    baseline_score = baseline.quality_score
                    baseline_row = baseline.to_dict()
                    rows.append(baseline_row)
                    query_aware = run_request(
                        request,
                        compressor=QueryAwareMMRCompressor(retain_ratio=0.50),
                        backend=backend,
                        tokenizer=tokenizer,
                        baseline_quality_score=baseline_score,
                        quality_fn=evaluate_real_request,
                    )
                    rows.append(query_aware.to_dict())
                    graph_row = run_request(
                        request,
                        compressor=graph_compressor,
                        backend=backend,
                        tokenizer=tokenizer,
                        baseline_quality_score=baseline_score,
                        quality_fn=evaluate_real_request,
                    )
                    rows.append(graph_row.to_dict())
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
                "sequence_id": sequence_id,
                "dataset": sequence_id.split("_sequence_", 1)[0],
                "request_ids": [request.request_id for request in sequence],
                "adjacent_shared_titles": [
                    sorted(_title_set(left) & _title_set(right))
                    for left, right in zip(sequence, sequence[1:])
                ],
            }
            for sequence_id, sequence in sequences
        ],
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "summaries": _summarize(rows),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
