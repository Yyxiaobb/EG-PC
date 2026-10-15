"""10% real Qwen 7B comparison for post-generation LLMLingua-2 memory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
from ladc.sequences import SequenceSpec, build_related_sequence_specs  # noqa: E402
from ladc.tokenizers import HuggingFaceTokenizer  # noqa: E402


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


def _annotate(request, spec: SequenceSpec, turn_index: int):
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
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))]


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        request_meta = row["metadata"]["request_metadata"]
        groups[(str(row["method"]), int(request_meta["turn_index"]))].append(row)
    summaries = []
    for (method, turn_index), group in sorted(groups.items()):
        metrics = [row.get("quality_metrics", {}) for row in group]
        compression = [row["metadata"].get("compression_metadata", {}) for row in group]
        latency = [float(row["latency"]["end_to_end_ms"]) for row in group]
        summaries.append({
            "method": method,
            "turn_index": turn_index,
            "n": len(group),
            "mean_input_tokens": sum(float(row["compressed_prompt_tokens"]) for row in group) / len(group),
            "mean_token_reduction_rate": sum(float(row["token_reduction_rate"]) for row in group) / len(group),
            "mean_token_f1": sum(float(metric.get("token_f1", 0.0)) for metric in metrics) / len(metrics),
            "mean_exact_match": sum(float(metric.get("exact_match", 0.0)) for metric in metrics) / len(metrics),
            "mean_evidence_f1": sum(float(metric.get("evidence_f1", 0.0)) for metric in metrics) / len(metrics),
            "mean_compression_ms": sum(float(row["latency"]["compression_ms"]) for row in group) / len(group),
            "mean_prefill_ms": sum(float(row["latency"]["prefill_ms"]) for row in group) / len(group),
            "mean_decode_ms": sum(float(row["latency"]["decode_ms"]) for row in group) / len(group),
            "mean_end_to_end_ms": sum(latency) / len(latency),
            "p50_end_to_end_ms": _percentile(latency, 0.5),
            "p95_end_to_end_ms": _percentile(latency, 0.95),
            "mean_reused_evidence_count": sum(float(meta.get("reused_evidence_count", 0.0)) for meta in compression) / len(compression),
            "mean_historical_candidate_count": sum(float(meta.get("historical_candidate_count", 0.0)) for meta in compression) / len(compression),
            "mean_history_compression_ms": sum(float(meta.get("history_compression_ms", 0.0)) for meta in compression) / len(compression),
            "history_completed_before_next_request_rate": sum(bool(meta.get("history_compression_completed_before_next_request", False)) for meta in compression) / len(compression),
            "history_pending_rate_at_record": sum(bool(meta.get("history_compression_pending", False)) for meta in compression) / len(compression),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\HF_dl\dataset"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/qwen2.5-7b-instruct"))
    parser.add_argument("--server", type=Path, default=Path("artifacts/llama_cpp/runtime/llama-server.exe"))
    parser.add_argument("--model", type=Path, default=Path(r"D:\HF_dl\model\Qwen2.5-7B-Instruct-GGUF\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"))
    parser.add_argument("--llmlingua-model", type=Path, default=Path(r"D:\HF_dl\model\llmlingua-2-bert-base-multilingual-cased-meetingbank"))
    parser.add_argument("--output", type=Path, default=Path("results/real_7b_post_generation_combo_10pct.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/real_7b_post_generation_combo_10pct_summary.json"))
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--max-sequences-per-dataset", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=9942)
    parser.add_argument("--inter-turn-wait-ms", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer = HuggingFaceTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    requests_by_dataset = _load_all_requests(args.data_root)
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    # Sequence construction is deterministic; cap by the requested fraction
    # while retaining at least one sequence per available dataset.
    sequences = build_related_sequence_specs(
        requests_by_dataset,
        sequence_length=args.sequence_length,
        max_sequences_per_dataset=args.max_sequences_per_dataset,
    )
    if args.fraction < 1.0:
        target = max(1, round(len(sequences) * args.fraction / 0.10)) if args.fraction < 0.10 else round(len(sequences) * args.fraction / 0.10)
        sequences = sequences[:max(1, min(len(sequences), target))]
    if not sequences:
        raise RuntimeError("no related sequences found")

    backend = LlamaServerInferenceBackend(
        server_path=args.server,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        port=args.port,
        max_new_tokens=args.max_new_tokens,
        extra_args=("--gpu-layers", "all", "--flash-attn", "on", "--ctx-size", "8192"),
        log_path="results/llama_server_post_generation_combo.log",
    )
    rows: list[dict[str, object]] = []
    combo: PostGenerationLLMLingua2MemoryCompressor | None = None
    pre_llm: LLMLingua2Compressor | None = None
    try:
        with backend:
            backend.generate("Answer with one word. Question: What is 2+2?", tokenizer=tokenizer)
            combo = PostGenerationLLMLingua2MemoryCompressor(
                str(args.llmlingua_model),
                retain_ratio=0.50,
                model_config={"local_files_only": True},
                name="evidence_graph_post_llmlingua2",
            )
            pre_llm = LLMLingua2Compressor(
                str(args.llmlingua_model),
                retain_ratio=0.50,
                model_config={"local_files_only": True},
            )
            for spec in sequences:
                session = SessionEvidenceGraphCompressor()
                previous_combo_row = None
                combo_rows: list[dict[str, object]] = []
                for turn_index, raw_request in enumerate(spec.requests):
                    request = _annotate(raw_request, spec, turn_index)
                    if previous_combo_row is not None:
                        if args.inter_turn_wait_ms > 0:
                            time.sleep(args.inter_turn_wait_ms / 1000.0)
                        combo.mark_completed_before_next_request(previous_combo_row["request_id"])
                        previous_combo_row["metadata"]["compression_metadata"].update(combo.history_audit(previous_combo_row["request_id"]))
                    baseline = run_request(request, compressor=NoCompressionCompressor(), backend=backend, tokenizer=tokenizer, quality_fn=evaluate_real_request)
                    baseline_score = baseline.quality_score
                    rows.append(baseline.to_dict())
                    for compressor in (session, pre_llm, combo):
                        result = run_request(request, compressor=compressor, backend=backend, tokenizer=tokenizer, baseline_quality_score=baseline_score, quality_fn=evaluate_real_request)
                        row = result.to_dict()
                        if compressor is combo:
                            row["metadata"]["compression_metadata"].update(combo.history_audit(request.request_id))
                            previous_combo_row = row
                            combo_rows.append(row)
                        rows.append(row)
                combo.wait_for_history()
                for combo_row in combo_rows:
                    request_id = combo_row["request_id"]
                    combo_row["metadata"]["compression_metadata"].update(combo.history_audit(request_id))
                session.close()
    finally:
        if combo is not None:
            combo.close()
        backend.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "rows": len(rows),
        "sequence_count": len(sequences),
        "sequence_length": args.sequence_length,
        "fraction": args.fraction,
        "inter_turn_wait_ms": args.inter_turn_wait_ms,
        "methods": ["none", "evidence_graph_session", "llmlingua2", "evidence_graph_post_llmlingua2"],
        "model": str(args.model),
        "llmlingua_model": str(args.llmlingua_model),
        "sequences": [{"sequence_id": spec.sequence_id, "dataset": spec.dataset, "request_ids": [r.request_id for r in spec.requests]} for spec in sequences],
        "summaries": _summarize(rows),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
