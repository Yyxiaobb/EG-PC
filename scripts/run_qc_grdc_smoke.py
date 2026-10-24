"""Deterministic selector pilot for QC-GRDC.

This is a compressor-level smoke experiment, not a 7B quality claim.  The
oracle backend returns the known answer only when the answer string survives in
the prompt, which measures evidence retention while keeping the pilot
independent of model weights and network access.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re

from ladc.compressors import (
    FusedEvidenceGraphCompressor,
    NoCompressionCompressor,
    QualityConstrainedGraphRateDistortionCompressor,
    SessionEvidenceGraphCompressor,
)
from ladc.experiments.runner import run_request
from ladc.records import InferenceRecord
from ladc.tokenizers import WhitespaceTokenizer
from ladc.workloads import WorkloadRequest


QUESTION_RE = re.compile(r"Question:\s*(.*?)\nAnswer:", re.IGNORECASE | re.DOTALL)


class OracleBackend:
    name = "oracle_prompt_retention"

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def generate(self, prompt: str, *, tokenizer: WhitespaceTokenizer) -> InferenceRecord:
        match = QUESTION_RE.search(prompt)
        question = match.group(1).strip() if match else ""
        answer = self.answers.get(question, "")
        output = answer if answer and answer.casefold() in prompt.casefold() else "[missing-evidence]"
        input_tokens = tokenizer.count_tokens(prompt)
        return InferenceRecord(
            output_text=output,
            input_tokens=input_tokens,
            output_tokens=tokenizer.count_tokens(output),
            prefill_ms=0.08 * input_tokens,
            decode_ms=1.0,
            metadata={"oracle_answer_present": output != "[missing-evidence]"},
        )


def oracle_quality(request: WorkloadRequest, _prompt: str, output: str) -> dict[str, float]:
    retained = float(output.casefold() == request.references[0].casefold()) if request.references else 0.0
    return {"primary_quality": retained, "exact_match": retained}


def make_request(session: str, turn: int, question: str, answer: str, anchor: str) -> WorkloadRequest:
    fillers = [
        f"Background passage {i} records a generic relation about {anchor}."
        for i in range(1, 15)
    ]
    if turn == 1:
        evidence = [
            f"{anchor} was led by Alice Carter.",
            "Alice Carter was born in Bristol.",
        ]
        title = "Expedition"
    elif turn == 2:
        evidence = ["Alice Carter later moved to London.", "The move happened after the expedition."]
        title = "Alice Carter"
    else:
        evidence = ["Alice Carter played violin.", "The instrument was used in local concerts."]
        title = "Alice Carter"
    context = "\n".join([f"[{title}] " + evidence[0], *evidence[1:], *fillers])
    prompt = (
        "Answer the question using only the retrieved context.\n"
        f"Context:\n{context}\nQuestion: {question}\nAnswer:"
    )
    return WorkloadRequest(
        f"{session}-t{turn}", prompt, question, context, (answer,),
        metadata={"dataset": "qc_smoke", "session_id": session, "turn_index": turn - 1},
    )


def build_sessions(count: int = 12) -> list[list[WorkloadRequest]]:
    sessions = []
    for index in range(count):
        session = f"smoke-{index:03d}"
        anchor = f"expedition {index}"
        sessions.append([
            make_request(session, 1, f"Where was the leader of the north {anchor} born?", "Bristol", anchor),
            make_request(session, 2, "Which city did Alice Carter later move to?", "London", anchor),
            make_request(session, 3, "What instrument did Alice Carter play?", "violin", anchor),
        ])
    return sessions


def main() -> None:
    tokenizer = WhitespaceTokenizer()
    answers = {
        "Where was the leader of the north expedition 0 born?": "Bristol",
        "Which city did Alice Carter later move to?": "London",
        "What instrument did Alice Carter play?": "violin",
    }
    # All sessions use the same question text for the oracle; the first turn's
    # anchor varies only in the evidence and does not change the answer.
    answers.update({
        f"Where was the leader of the north expedition {i} born?": "Bristol"
        for i in range(12)
    })
    backend = OracleBackend(answers)
    rows = []
    for method in ("no_compression", "evidence_graph_fused_v2", "evidence_graph_qc_grdc_v4"):
        for sequence in build_sessions():
            if method == "no_compression":
                compressor = NoCompressionCompressor()
            elif method == "evidence_graph_fused_v2":
                compressor = SessionEvidenceGraphCompressor(compressor_class=FusedEvidenceGraphCompressor)
            else:
                compressor = SessionEvidenceGraphCompressor(
                    compressor_class=QualityConstrainedGraphRateDistortionCompressor,
                    compressor_kwargs={"quality_floor": 0.92, "max_hops": 3},
                )
            try:
                for request in sequence:
                    row = run_request(
                        request,
                        compressor=compressor,
                        backend=backend,
                        tokenizer=tokenizer,
                        target_budget_ratio=None if method == "no_compression" else 0.55,
                        quality_fn=oracle_quality,
                    ).to_dict()
                    row["method"] = method
                    rows.append(row)
                if hasattr(compressor, "wait_for_history"):
                    compressor.wait_for_history()
            finally:
                if hasattr(compressor, "close"):
                    compressor.close()

    summary = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    for method, items in grouped.items():
        summary.append({
            "method": method,
            "requests": len(items),
            "oracle_evidence_recall": sum(
                r["metadata"]["output_text"] != "[missing-evidence]" for r in items
            ) / len(items),
            "avg_input_tokens": sum(r["compressed_prompt_tokens"] for r in items) / len(items),
            "token_reduction_rate": sum(r["token_reduction_rate"] for r in items) / len(items),
            "avg_end_to_end_ms": sum(r["latency"]["end_to_end_ms"] for r in items) / len(items),
            "avg_compression_wallclock_ms": sum(r["metadata"]["compression_wallclock_ms"] for r in items) / len(items),
            "quality_proxy_ratio": sum(float(r["metadata"]["compression_metadata"].get("quality_proxy_ratio", 1.0)) for r in items) / len(items),
        })
    output = Path("results/qc_grdc_smoke_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": "12 synthetic 3-turn sessions; oracle evidence retention", "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
