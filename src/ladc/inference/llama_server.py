"""Persistent llama.cpp server backend for fair per-request measurements."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from ladc.interfaces import Tokenizer
from ladc.records import InferenceRecord


def _extract_timings(payload: dict[str, Any]) -> tuple[float, float]:
    timings = payload.get("timings") or {}
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")
    if prompt_ms is None or predicted_ms is None:
        raise RuntimeError("llama-server response did not contain prompt_ms and predicted_ms timings")
    return float(prompt_ms), float(predicted_ms)


class LlamaServerInferenceBackend:
    """Keep one llama.cpp process alive and issue local completion requests."""

    name = "llama_server"

    def __init__(
        self,
        *,
        server_path: str | Path,
        model_path: str | Path,
        tokenizer_path: str | Path,
        port: int = 9931,
        max_new_tokens: int = 64,
        system_prompt: str = "You are a concise question answering assistant.",
        startup_timeout_s: float = 180.0,
        extra_args: Sequence[str] = (),
        log_path: str | Path = "results/llama_server.log",
    ) -> None:
        # llama.cpp uses -1 for generation until EOS (or the context limit).
        # Keep positive values available for bounded runs, but reject zero and
        # values below the supported unlimited sentinel.
        if max_new_tokens == 0 or max_new_tokens < -1:
            raise ValueError("max_new_tokens must be positive or -1 for no fixed generation limit")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        self.server_path = str(server_path)
        self.model_path = str(model_path)
        self.tokenizer_path = str(tokenizer_path)
        self.port = port
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.startup_timeout_s = startup_timeout_s
        self.extra_args = tuple(str(arg) for arg in extra_args)
        self.log_path = Path(log_path)
        self._process: subprocess.Popen[str] | None = None
        self._log_handle = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        command = [
            self.server_path,
            "-m",
            self.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--no-warmup",
            "--metrics",
            "--jinja",
            *self.extra_args,
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup; see {self.log_path}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.5)
        raise TimeoutError(f"llama-server did not become healthy within {self.startup_timeout_s}s")

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._process = None

    def __enter__(self) -> "LlamaServerInferenceBackend":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _post_completion(self, formatted_prompt: str) -> dict[str, Any]:
        body = json.dumps(
            {
                "prompt": formatted_prompt,
                "n_predict": self.max_new_tokens,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "stream": False,
                "cache_prompt": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(self, prompt: str, *, tokenizer: Tokenizer) -> InferenceRecord:
        if self._process is None or self._process.poll() is not None:
            self.start()
        if hasattr(tokenizer, "format_chat_prompt"):
            formatted_prompt = tokenizer.format_chat_prompt(prompt, system_prompt=self.system_prompt)
            input_tokens = tokenizer.count_chat_tokens(prompt, system_prompt=self.system_prompt)
        else:
            formatted_prompt = prompt
            input_tokens = tokenizer.count_tokens(prompt)
        started = time.perf_counter()
        payload = self._post_completion(formatted_prompt)
        wallclock_ms = (time.perf_counter() - started) * 1000.0
        prefill_ms, decode_ms = _extract_timings(payload)
        output_text = str(payload.get("content", "")).strip()
        usage = payload.get("usage") or {}
        output_tokens = int(usage.get("completion_tokens", tokenizer.count_tokens(output_text)))
        return InferenceRecord(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            metadata={
                "backend": self.name,
                "server_path": self.server_path,
                "model_path": self.model_path,
                "tokenizer_path": self.tokenizer_path,
                "wallclock_ms": wallclock_ms,
                "timing_source": "llama-server response timings",
                "server_port": self.port,
                "raw_timings": payload.get("timings", {}),
            },
        )
