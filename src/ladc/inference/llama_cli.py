"""Real llama.cpp CLI inference backend with explicit latency accounting."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Sequence

from ladc.interfaces import Tokenizer
from ladc.records import InferenceRecord


_PROMPT_EVAL_RE = re.compile(r"prompt\s+eval\s+time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
_DECODE_EVAL_RE = re.compile(
    r"^(?!.*prompt\s+eval\s+time).*?(?:\beval|\bdecode)\s+time\s*=\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*ms",
    re.IGNORECASE | re.MULTILINE,
)
_THROUGHPUT_RE = re.compile(
    r"Prompt:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s\s*\|\s*"
    r"Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s",
    re.IGNORECASE,
)


def _parse_latency(stderr: str, stdout: str) -> tuple[float, float]:
    """Parse llama.cpp performance lines without guessing missing components."""

    combined = f"{stderr}\n{stdout}"
    prompt_match = _PROMPT_EVAL_RE.search(combined)
    decode_match = _DECODE_EVAL_RE.search(combined)
    if prompt_match is None or decode_match is None:
        raise RuntimeError(
            "llama.cpp output did not contain both prompt eval time and eval time; "
            "refusing to fabricate prefill/decode latency"
        )
    return float(prompt_match.group(1)), float(decode_match.group(1))


def _parse_throughput(stderr: str, stdout: str) -> tuple[float, float]:
    match = _THROUGHPUT_RE.search(f"{stderr}\n{stdout}")
    if match is None:
        raise RuntimeError(
            "llama.cpp output contained neither millisecond timings nor prompt/generation throughput"
        )
    return float(match.group(1)), float(match.group(2))


def _clean_generation(stdout: str) -> str:
    """Remove common llama.cpp performance lines if a build writes them to stdout."""

    lines = stdout.splitlines()
    prompt_echo = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("> ")),
        None,
    )
    if prompt_echo is not None:
        lines = lines[prompt_echo + 1 :]
    for index, line in enumerate(lines):
        if "[ Prompt:" in line:
            lines = lines[:index]
            break

    ignored_prefixes = (
        "llama_perf_",
        "prompt eval time",
        "eval time",
        "total time",
        "load time",
    )
    lines = [line for line in lines if not line.strip().lower().startswith(ignored_prefixes)]
    lines = [line for line in lines if line.strip().lower() not in {"exiting...", ">"}]
    return "\n".join(lines).strip()


class LlamaCliInferenceBackend:
    """Run one prompt through a llama.cpp CLI executable.

    The backend assumes a llama.cpp build that accepts ``-m``, ``-p``, ``-n``
    and ``--no-display-prompt`` and emits standard performance summaries. Extra
    flags can be supplied for GPU layers, threads, batch size, or sampling.
    """

    name = "llama_cli"

    def __init__(
        self,
        *,
        cli_path: str | Path,
        model_path: str | Path,
        max_new_tokens: int = 64,
        timeout_s: float = 600.0,
        system_prompt: str | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.cli_path = str(cli_path)
        self.model_path = str(model_path)
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt
        self.extra_args = tuple(str(arg) for arg in extra_args)

    def _command(self, prompt: str) -> list[str]:
        return [
            self.cli_path,
            "-m",
            self.model_path,
            "-p",
            prompt,
            "-n",
            str(self.max_new_tokens),
            "--no-display-prompt",
            "--single-turn",
            "--no-warmup",
            "--perf",
            "--jinja",
            "--simple-io",
            *( ["--system-prompt", self.system_prompt] if self.system_prompt else [] ),
            *self.extra_args,
        ]

    def generate(self, prompt: str, *, tokenizer: Tokenizer) -> InferenceRecord:
        command = self._command(prompt)
        started = perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"llama.cpp CLI not found: {self.cli_path}") from exc
        wallclock_ms = (perf_counter() - started) * 1000.0
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp exited with code {completed.returncode}: "
                f"{completed.stderr[-1000:]}"
            )
        output_text = _clean_generation(completed.stdout)
        input_tokens = (
            tokenizer.count_chat_tokens(prompt, system_prompt=self.system_prompt)
            if self.system_prompt and hasattr(tokenizer, "count_chat_tokens")
            else tokenizer.count_tokens(prompt)
        )
        output_tokens = tokenizer.count_tokens(output_text)
        try:
            prefill_ms, decode_ms = _parse_latency(completed.stderr, completed.stdout)
            timing_source = "llama.cpp millisecond performance summary"
        except RuntimeError:
            prompt_tps, generation_tps = _parse_throughput(completed.stderr, completed.stdout)
            prefill_ms = 1000.0 * input_tokens / prompt_tps
            decode_ms = 1000.0 * output_tokens / generation_tps if output_tokens else 0.0
            timing_source = "llama.cpp throughput summary converted with native token counts"
        return InferenceRecord(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            metadata={
                "backend": self.name,
                "cli_path": self.cli_path,
                "model_path": self.model_path,
                "wallclock_ms": wallclock_ms,
                "timing_source": timing_source,
                "prompt_tokens_per_second": 1000.0 * input_tokens / prefill_ms if prefill_ms else None,
                "generation_tokens_per_second": 1000.0 * output_tokens / decode_ms if decode_ms else None,
                "returncode": completed.returncode,
            },
        )
