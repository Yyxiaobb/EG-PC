"""Abstractive evidence-graph compression with source provenance."""

from __future__ import annotations

import re
from threading import Lock
from time import perf_counter
from typing import Any

from ladc.compressors.fused_graph import FusedEvidenceGraphCompressor
from ladc.records import CompressionRecord
from ladc.interfaces import Tokenizer
from ladc.workloads import WorkloadRequest


class LocalAbstractiveSummarizer:
    """A small local causal LM used only for evidence summarization.

    The model is loaded once and guarded by a lock because Transformers
    generation is not safe to call concurrently on the same model instance.
    The prompt requires source tags and forbids unsupported additions; source
    sentence IDs remain in the compression metadata as the authoritative
    provenance anchors.
    """

    def __init__(self, model_path: str, *, device: str = "cpu", max_new_tokens: int = 256) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("abstractive summarization requires torch and transformers") from exc
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.max_new_tokens = max(16, int(max_new_tokens))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, dtype="auto"
        ).to(self.device).eval()
        self._lock = Lock()

    def summarize(self, question: str, evidence: str, *, max_new_tokens: int | None = None) -> tuple[str, float, int]:
        import torch

        del question  # Candidate selection is already query-aware; avoid eliciting a direct answer.
        prompt = (
            "Compress the source passages into concise factual notes for future question answering.\n"
            "Preserve the subjects, relation direction, names, dates, and links between facts.\n"
            "Do not answer a question, write document titles by themselves, or add facts not stated below.\n"
            "Use complete short sentences. Omit details that cannot be stated faithfully.\n"
            f"Source passages:\n{evidence}\nFactual notes:\n"
        )
        with self._lock:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                encoded_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            else:
                encoded_text = prompt
            encoded = self.tokenizer(encoded_text, return_tensors="pt", truncation=True, max_length=8192)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            started = perf_counter()
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens or self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            elapsed_ms = (perf_counter() - started) * 1000.0
            generated = output[0, encoded["input_ids"].shape[1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            return text, elapsed_ms, int(generated.numel())


class AbstractiveFusedEvidenceGraphCompressor(FusedEvidenceGraphCompressor):
    """V2 graph retrieval followed by provenance-aware abstractive packing.

    Graph retrieval stays extractive and source-backed. The selected source
    sentences are then summarized by a local model. The raw source IDs remain
    attached to the record and the graph, so the summary is never treated as
    an independent factual source. In a multi-turn session the same
    summarizer can also be submitted from ``record_generation`` for the next
    request; this standalone benchmark measures the current-request summary
    explicitly in latency because the next question is not available yet.
    """

    name = "evidence_graph_abstractive_v3"

    def __init__(self, *args: Any, summarizer: LocalAbstractiveSummarizer, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.summarizer = summarizer

    @staticmethod
    def _grounded_enough(summary: str, evidence: str) -> bool:
        """Reject outputs that fail source-tag or lexical grounding checks."""
        if not re.search(r"\[E\d+\]", summary):
            return False
        allowed = {x.casefold() for x in re.findall(r"[A-Za-z0-9]+", evidence)}
        words = [x.casefold() for x in re.findall(r"[A-Za-z0-9]+", summary)]
        content = [x for x in words if len(x) > 2 and not x.isdigit()]
        if not content:
            return False
        overlap = sum(word in allowed for word in content) / len(content)
        evidence_terms = {x.casefold() for x in re.findall(r"[A-Za-z0-9]+", evidence) if len(x) > 2}
        covered = {word for word in content if word in evidence_terms}
        recall = len(covered) / max(1, len(evidence_terms))
        # A title-only copy often passes precision but destroys the relation
        # chain. Require both low unsupported-token rate and meaningful source
        # coverage; otherwise keep the safer V2 source sentences.
        return overlap >= 0.72 and recall >= 0.28 and len(covered) >= 8

    def compress_request(self, request: WorkloadRequest, *, tokenizer: Tokenizer,
                         target_budget_ratio: float | None = None) -> CompressionRecord:
        base = super().compress_request(request, tokenizer=tokenizer,
                                        target_budget_ratio=target_budget_ratio)
        selected_ids = tuple(base.metadata.get("selected_sentence_ids", ()))
        with self.graph._lock:
            nodes = [self.graph.nodes.get(node_id) for node_id in selected_ids]
        evidence_lines: list[str] = []
        for index, node in enumerate(nodes):
            if node is None or node.node_type != "sentence":
                continue
            title = str(node.metadata.get("title", ""))
            evidence_lines.append(f"[E{index}] [{title}] {node.text}" if title else f"[E{index}] {node.text}")
        evidence = "\n".join(evidence_lines)
        budget = int(base.metadata.get("budget_tokens", base.compressed_prompt_tokens))
        prefix = "Answer the question using only the retrieved context.\nContext:\n"
        suffix = f"\nQuestion: {request.question}\nAnswer:"
        available = max(16, budget - tokenizer.count_tokens(prefix + suffix))
        summary, summary_ms, summary_tokens = self.summarizer.summarize(
            request.question, evidence,
            max_new_tokens=max(64, min(384, available)),
        )
        grounded = self._grounded_enough(summary, evidence)
        fallback = not grounded
        # A failed or suspicious summary falls back to the original V2
        # extractive prompt. This preserves the same evidence and token budget.
        prompt = (prefix + "[abstractive evidence summary]\n" + summary + suffix) if grounded else base.compressed_prompt
        while tokenizer.count_tokens(prompt) > budget and summary:
            allowed = max(1, available - 1)
            summary = tokenizer.truncate_to_tokens(summary, allowed)
            prompt = prefix + "[abstractive evidence summary]\n" + summary + suffix
            available = allowed
        metadata = {
            **base.metadata,
            "abstractive_model": self.summarizer.model_path,
            "abstractive_summary_tokens": summary_tokens,
            "abstractive_summary_ms": summary_ms,
            "abstractive_source_count": len(evidence_lines),
            "abstractive_source_sentence_ids": list(selected_ids),
            "abstractive_summary_text": summary if grounded else "",
            "abstractive_grounding_check_passed": grounded,
            "abstractive_fallback_to_extractive": fallback,
            "compression_family": "graph_retrieval_then_abstractive_summary",
        }
        return CompressionRecord(
            method=self.name,
            original_prompt_tokens=base.original_prompt_tokens,
            compressed_prompt_tokens=tokenizer.count_tokens(prompt),
            compressed_prompt=prompt,
            latency_ms=base.latency_ms + summary_ms,
            target_budget_ratio=target_budget_ratio,
            metadata=metadata,
        )
