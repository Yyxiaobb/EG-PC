"""Run compressors and inference backends under one accounting protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from time import perf_counter

from ladc.controller import BudgetCandidate, BudgetDecision, DynamicBudgetController
from ladc.interfaces import InferenceBackend, PromptCompressor, Tokenizer
from ladc.records import LatencyBreakdown, RequestRecord
from ladc.workloads import SyntheticRequest, WorkloadRequest


Request = SyntheticRequest | WorkloadRequest
QualityFn = Callable[[Request, str, str], dict[str, float]]


def run_request(
    request: Request,
    *,
    compressor: PromptCompressor,
    backend: InferenceBackend,
    tokenizer: Tokenizer,
    baseline_quality_score: float | None = None,
    quality_fn: QualityFn | None = None,
    target_budget_ratio: float | None = None,
) -> RequestRecord:
    """Run one request and include compressor overhead in end-to-end latency."""

    request_started = perf_counter()
    request_compressor = getattr(compressor, "compress_request", None)
    if callable(request_compressor):
        compressed = request_compressor(
            request,
            tokenizer=tokenizer,
            target_budget_ratio=target_budget_ratio,
        )
    else:
        compressed = compressor.compress(
            request.prompt,
            tokenizer=tokenizer,
            target_budget_ratio=target_budget_ratio,
        )
    compression_wallclock_ms = (perf_counter() - request_started) * 1000
    inference = backend.generate(compressed.compressed_prompt, tokenizer=tokenizer)
    request_wallclock_ms = (perf_counter() - request_started) * 1000
    # Evidence-graph compressors persist a compact Q&A memory after the model
    # has produced its answer.  The hook is non-blocking, so this work is not
    # included in the user-visible inference timing below.
    history_hook = getattr(compressor, "record_generation", None)
    if callable(history_hook) and isinstance(request, WorkloadRequest):
        history_hook(
            request,
            inference.output_text,
            compression=compressed,
            tokenizer=tokenizer,
        )
    quality_metrics = (
        {}
        if quality_fn is None
        else quality_fn(request, compressed.compressed_prompt, inference.output_text)
    )
    quality_score = quality_metrics.get("primary_quality")
    return RequestRecord(
        request_id=request.request_id,
        method=compressed.method,
        original_prompt_tokens=compressed.original_prompt_tokens,
        compressed_prompt_tokens=compressed.compressed_prompt_tokens,
        output_tokens=inference.output_tokens,
        quality_score=quality_score,
        baseline_quality_score=baseline_quality_score,
        target_budget_ratio=target_budget_ratio,
        quality_metrics=quality_metrics,
        latency=LatencyBreakdown(
            compression_ms=compressed.latency_ms,
            prefill_ms=inference.prefill_ms,
            decode_ms=inference.decode_ms,
        ),
        metadata={
            "backend": backend.name,
            "request_metadata": request.metadata,
            "compression_metadata": compressed.metadata,
            "inference_metadata": inference.metadata,
            "request_wallclock_ms": request_wallclock_ms,
            "compression_wallclock_ms": compression_wallclock_ms,
            "output_text": inference.output_text,
            "compressed_prompt": compressed.compressed_prompt,
        },
    )


def run_workload_matrix(
    requests: Iterable[Request],
    *,
    compressors: Iterable[PromptCompressor],
    backend: InferenceBackend,
    tokenizer: Tokenizer,
    target_budget_ratio: float | None = None,
    quality_fn: QualityFn | None = None,
) -> list[RequestRecord]:
    """Run every compressor on every request in deterministic order."""

    rows: list[RequestRecord] = []
    for request in requests:
        for compressor in compressors:
            rows.append(
                run_request(
                    request,
                    compressor=compressor,
                    backend=backend,
                    tokenizer=tokenizer,
                    target_budget_ratio=target_budget_ratio,
                    quality_fn=quality_fn,
                )
            )
    return rows


def run_dynamic_request(
    request: Request,
    *,
    candidates: list[BudgetCandidate],
    compressors: dict[str, PromptCompressor],
    controller: DynamicBudgetController,
    backend: InferenceBackend,
    tokenizer: Tokenizer,
    quality_fn: QualityFn | None = None,
) -> tuple[RequestRecord, BudgetDecision]:
    """Select one compression plan, then execute only that plan."""

    decision = controller.choose(
        original_prompt_tokens=tokenizer.count_tokens(request.prompt),
        output_tokens=request.target_output_tokens,
        candidates=candidates,
    )
    candidate = decision.candidate
    if candidate.method not in compressors:
        raise KeyError(f"no compressor registered for method: {candidate.method}")
    row = run_request(
        request,
        compressor=compressors[candidate.method],
        backend=backend,
        tokenizer=tokenizer,
        quality_fn=quality_fn,
        target_budget_ratio=candidate.target_budget_ratio,
    )
    row = RequestRecord(
        request_id=row.request_id,
        method="dynamic:" + row.method,
        original_prompt_tokens=row.original_prompt_tokens,
        compressed_prompt_tokens=row.compressed_prompt_tokens,
        output_tokens=row.output_tokens,
        quality_score=row.quality_score,
        baseline_quality_score=row.baseline_quality_score,
        latency=row.latency,
        target_budget_ratio=row.target_budget_ratio,
        quality_metrics=row.quality_metrics,
        metadata={**row.metadata, "budget_decision": decision.to_dict()},
    )
    return row, decision
