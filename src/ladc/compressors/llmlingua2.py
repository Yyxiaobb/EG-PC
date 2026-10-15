"""Optional LLMLingua-2 prompt compression.

The dependency and model are loaded only when this compressor is constructed.
This keeps the core package usable in environments that do not install the
large Transformer stack.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ladc.compressors.evidence_graph import _request_from_prompt
from ladc.interfaces import Tokenizer
from ladc.records import CompressionRecord
from ladc.workloads import WorkloadRequest


class LLMLingua2Compressor:
    """Compress retrieved context with a local LLMLingua-2 model."""

    name = "llmlingua2"

    def __init__(
        self,
        model_path: str,
        *,
        retain_ratio: float = 0.50,
        device_map: str = "cuda",
        model_config: dict[str, Any] | None = None,
        llmlingua2_config: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        if not 0.0 < retain_ratio <= 1.0:
            raise ValueError("retain_ratio must be in the interval (0, 1]")
        try:
            from llmlingua import PromptCompressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LLMLingua-2 requires the optional llmlingua package") from exc

        self.retain_ratio = retain_ratio
        self.model_path = str(model_path)
        self._compressor = PromptCompressor(
            self.model_path,
            device_map=device_map,
            model_config=dict(model_config or {}),
            use_llmlingua2=True,
            llmlingua2_config=dict(llmlingua2_config or {}),
        )
        if name is not None:
            self.name = name

    def compress(
        self,
        prompt: str,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        return self.compress_request(
            _request_from_prompt(prompt),
            tokenizer=tokenizer,
            target_budget_ratio=target_budget_ratio,
        )

    def compress_request(
        self,
        request: WorkloadRequest,
        *,
        tokenizer: Tokenizer,
        target_budget_ratio: float | None = None,
    ) -> CompressionRecord:
        ratio = self.retain_ratio if target_budget_ratio is None else target_budget_ratio
        if not 0.0 < ratio <= 1.0:
            raise ValueError("target budget ratio must be in the interval (0, 1]")
        original_tokens = tokenizer.count_tokens(request.prompt)
        context_segments: list[str] = []
        # LLMLingua-2's token classifier has a 512-token sequence window.
        # Split unusually long dataset paragraphs before handing them to the
        # library so a single paragraph cannot overflow that window.
        for line in (line.strip() for line in request.context.splitlines() if line.strip()):
            words = line.split()
            if len(words) <= 120:
                context_segments.append(line)
                continue
            context_segments.extend(
                " ".join(words[index : index + 120])
                for index in range(0, len(words), 120)
            )
        if not context_segments:
            context_segments = [request.context.strip()] if request.context.strip() else [""]

        started = perf_counter()
        result = self._compressor.compress_prompt(
            context_segments,
            question=request.question,
            rate=ratio,
            keep_split=True,
            strict_preserve_uncompressed=True,
        )
        compressed_context = str(result.get("compressed_prompt", "")).strip()
        compressed_prompt = (
            "Answer the question using only the retrieved context.\n"
            f"Context:\n{compressed_context}\n"
            f"Question: {request.question}\nAnswer:"
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        return CompressionRecord(
            method=self.name,
            original_prompt_tokens=original_tokens,
            compressed_prompt_tokens=tokenizer.count_tokens(compressed_prompt),
            compressed_prompt=compressed_prompt,
            latency_ms=elapsed_ms,
            target_budget_ratio=target_budget_ratio,
            metadata={
                "llmlingua2_model": self.model_path,
                "llmlingua2_rate": ratio,
                "llmlingua2_origin_tokens": result.get("origin_tokens"),
                "llmlingua2_compressed_tokens": result.get("compressed_tokens"),
                "llmlingua2_ratio": result.get("ratio"),
                "llmlingua2_rate_reported": result.get("rate"),
                "llmlingua2_saving": result.get("saving"),
            },
        )
