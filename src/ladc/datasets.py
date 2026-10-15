"""Read-only adapters for public long-context QA datasets.

The loaders deliberately keep evaluation annotations separate from ``prompt``.
They accept common exported formats (JSONL, JSON, Parquet and Arrow) so the
experiment can use an existing local dataset copy without preprocessing it
into a project-specific format.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from random import Random
from typing import Any

from ladc.workloads import WorkloadRequest


def _limit_reached(count: int, max_examples: int | None) -> bool:
    return max_examples is not None and count >= max_examples


def _candidate_files(path: str | Path, suffixes: set[str], split: str | None) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"dataset path does not exist: {root}")
    files = sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in suffixes)
    if not files:
        raise FileNotFoundError(f"no supported dataset files found below: {root}")
    # HuggingFace dataset directories contain JSON metadata next to the real
    # Arrow/Parquet shards. Prefer data shards so ``dataset_info.json`` cannot
    # be mistaken for a list of examples.
    preferred_suffixes = [".parquet", ".arrow", ".jsonl", ".json"]
    for preferred_suffix in preferred_suffixes:
        preferred = [item for item in files if item.suffix.lower() == preferred_suffix]
        if preferred:
            files = preferred
            break
    if split:
        split_files = [item for item in files if split.lower() in item.stem.lower()]
        if split_files:
            files = split_files
    return files


def _iter_rows(path: str | Path, *, split: str | None = None) -> Iterator[dict[str, Any]]:
    suffixes = {".jsonl", ".json", ".parquet", ".arrow"}
    for file_path in _candidate_files(path, suffixes, split):
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"expected an object in {file_path}:{line_number}")
                    yield row
            continue
        if suffix == ".json":
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload if isinstance(payload, list) else payload.get("data", payload)
            if not isinstance(rows, list):
                raise ValueError(f"expected a list of records in {file_path}")
            yield from (row for row in rows if isinstance(row, dict))
            continue
        yield from _iter_arrow_rows(file_path)


def _iter_arrow_rows(file_path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Reading Parquet/Arrow requires pyarrow; install it in the experiment environment"
        ) from exc

    if file_path.suffix.lower() == ".parquet":
        table = parquet.read_table(file_path)
    else:
        try:
            with file_path.open("rb") as handle:
                table = ipc.open_file(handle).read_all()
        except (pa.ArrowInvalid, pa.ArrowIOError, ValueError):
            with file_path.open("rb") as handle:
                table = ipc.open_stream(handle).read_all()
    yield from table.to_pylist()


def _first(row: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return default


def _first_non_empty(row: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    """Return the first present value that is not an empty collection/string."""

    fallback = default
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        fallback = value
        if value != "" and value != [] and value != {}:
            return value
    return fallback


def _as_references(row: Mapping[str, Any]) -> tuple[str, ...]:
    answer = _first(row, "answer", "answers", "label", default="")
    aliases = _first(row, "answer_aliases", "aliases", default=[])
    values: list[str] = []
    if isinstance(answer, list):
        values.extend(str(item) for item in answer if str(item).strip())
    elif str(answer).strip():
        values.append(str(answer))
    if isinstance(aliases, list):
        values.extend(str(item) for item in aliases if str(item).strip())
    return tuple(dict.fromkeys(values))


def _format_context(context: Any) -> tuple[str, tuple[str, ...]]:
    """Normalize common title/sentence and paragraph representations."""

    blocks: list[str] = []
    titles: list[str] = []
    if isinstance(context, Mapping):
        # 2WikiMultihopQA stores one row as parallel ``title`` and
        # ``sentences`` columns rather than a list of document records.
        titles_column = context.get("title")
        sentences_column = context.get("sentences")
        if isinstance(titles_column, list) and isinstance(sentences_column, list):
            context = list(zip(titles_column, sentences_column))
        else:
            context = list(context.items())
    if isinstance(context, list):
        for index, item in enumerate(context):
            title = f"Document {index}"
            text: Any = item
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, text = item[0], item[1]
            elif isinstance(item, Mapping):
                title = _first(item, "title", "name", default=title)
                text = _first(item, "paragraph_text", "text", "sentences", "content", default="")
            title = str(title)
            if isinstance(text, list):
                text = " ".join(str(sentence) for sentence in text)
            text = str(text).strip()
            if text:
                titles.append(title)
                blocks.append(f"[{title}] {text}")
    elif context is not None:
        blocks.append(str(context).strip())
    return "\n".join(blocks), tuple(titles)


def _supporting_facts(value: Any) -> tuple[tuple[str, int], ...]:
    facts: list[tuple[str, int]] = []
    if isinstance(value, Mapping):
        titles_column = value.get("title")
        indexes_column = value.get("sent_id", value.get("sentence_id"))
        if isinstance(titles_column, list) and isinstance(indexes_column, list):
            value = list(zip(titles_column, indexes_column))
        else:
            value = value.get("facts", value.get("evidence", []))
    if not isinstance(value, list):
        return ()
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                facts.append((str(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue
        elif isinstance(item, Mapping):
            title = _first(item, "title", "document", "name", default="")
            index = _first(item, "sentence_id", "sentence_idx", "paragraph_id", "index", default=0)
            try:
                facts.append((str(title), int(index)))
            except (TypeError, ValueError):
                continue
    return tuple(facts)


def _make_request(
    row: Mapping[str, Any],
    *,
    dataset: str,
    index: int,
    split: str | None,
    context: Any,
    supporting: Any,
) -> WorkloadRequest:
    question = str(_first(row, "question", "query", default=""))
    context_text, titles = _format_context(context)
    prompt = (
        "Answer the question using only the retrieved context.\n"
        f"Context:\n{context_text}\n"
        f"Question: {question}\nAnswer:"
    )
    request_id = str(_first(row, "id", "_id", "question_id", default=f"{dataset}_{index}"))
    references = _as_references(row)
    return WorkloadRequest(
        request_id=request_id,
        prompt=prompt,
        question=question,
        context=context_text,
        references=references,
        supporting_facts=_supporting_facts(supporting),
        target_output_tokens=32,
        metadata={
            "dataset": dataset,
            "split": split,
            "context_document_count": len(titles),
            "context_titles": titles,
            "answerable": _first(row, "answerable", default=True),
        },
    )


def load_hotpotqa(
    path: str | Path,
    *,
    split: str | None = None,
    max_examples: int | None = None,
) -> list[WorkloadRequest]:
    """Load HotpotQA rows from an existing local export."""

    requests: list[WorkloadRequest] = []
    for index, row in enumerate(_iter_rows(path, split=split)):
        if _limit_reached(len(requests), max_examples):
            break
        requests.append(
            _make_request(
                row,
                dataset="hotpotqa",
                index=index,
                split=split,
                context=_first(row, "context", "documents", default=[]),
                supporting=_first(row, "supporting_facts", "evidence", default=[]),
            )
        )
    return requests


def load_2wikimultihopqa(
    path: str | Path,
    *,
    split: str | None = None,
    max_examples: int | None = None,
) -> list[WorkloadRequest]:
    """Load 2WikiMultihopQA rows from an existing local export."""

    requests: list[WorkloadRequest] = []
    for index, row in enumerate(_iter_rows(path, split=split)):
        if _limit_reached(len(requests), max_examples):
            break
        requests.append(
            _make_request(
                row,
                dataset="2wikimultihopqa",
                index=index,
                split=split,
                context=_first(row, "context", "documents", "paragraphs", default=[]),
                supporting=_first_non_empty(row, "supporting_facts", "evidences", "evidence", default=[]),
            )
        )
    return requests


def load_musique(
    path: str | Path,
    *,
    split: str | None = None,
    max_examples: int | None = None,
) -> list[WorkloadRequest]:
    """Load MuSiQue JSONL/JSON/Arrow rows from an existing local export."""

    requests: list[WorkloadRequest] = []
    for index, row in enumerate(_iter_rows(path, split=split)):
        if _limit_reached(len(requests), max_examples):
            break
        paragraphs = _first(row, "paragraphs", "context", default=[])
        supporting = [
            {"title": item.get("title", ""), "paragraph_id": item.get("idx", item.get("paragraph_id", 0))}
            for item in paragraphs
            if isinstance(item, Mapping) and item.get("is_supporting")
        ] if isinstance(paragraphs, list) else []
        requests.append(
            _make_request(
                row,
                dataset="musique",
                index=index,
                split=split,
                context=paragraphs,
                supporting=supporting,
            )
        )
    return requests


def length_bucket(token_count: int) -> str:
    """Return the standard reporting bucket for prompt length."""

    if token_count < 2048:
        return "<2K"
    if token_count < 4096:
        return "2K-4K"
    if token_count < 8192:
        return "4K-8K"
    if token_count < 16384:
        return "8K-16K"
    return "16K+"


def build_length_manifest(requests: list[WorkloadRequest], *, tokenizer) -> list[dict[str, Any]]:
    """Build evaluation metadata without serializing prompts or gold labels."""

    manifest: list[dict[str, Any]] = []
    for request in requests:
        prompt_tokens = tokenizer.count_tokens(request.prompt)
        manifest.append(
            {
                "request_id": request.request_id,
                "dataset": request.metadata.get("dataset"),
                "split": request.metadata.get("split"),
                "prompt_tokens": prompt_tokens,
                "length_bucket": length_bucket(prompt_tokens),
                "reference_count": len(request.references),
                "supporting_fact_count": len(request.supporting_facts),
                "context_document_count": request.metadata.get("context_document_count", 0),
            }
        )
    return manifest


def select_balanced_requests(
    requests: list[WorkloadRequest],
    *,
    tokenizer,
    max_per_bucket: int,
    seed: int = 42,
    require_references: bool = True,
) -> list[WorkloadRequest]:
    """Select a deterministic, length-balanced subset for expensive inference."""

    if max_per_bucket <= 0:
        raise ValueError("max_per_bucket must be positive")
    groups: dict[str, list[WorkloadRequest]] = {}
    for request in requests:
        if require_references and not request.references:
            continue
        bucket = length_bucket(tokenizer.count_tokens(request.prompt))
        groups.setdefault(bucket, []).append(request)

    rng = Random(seed)
    selected: list[WorkloadRequest] = []
    bucket_order = ["<2K", "2K-4K", "4K-8K", "8K-16K", "16K+"]
    for bucket in bucket_order:
        candidates = sorted(groups.get(bucket, []), key=lambda request: request.request_id)
        rng.shuffle(candidates)
        selected.extend(candidates[:max_per_bucket])
    return selected
