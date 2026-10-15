from unittest.mock import patch

import pytest

from ladc.inference.llama_cli import (
    LlamaCliInferenceBackend,
    _clean_generation,
    _parse_latency,
    _parse_throughput,
)
from ladc.tokenizers import WhitespaceTokenizer


def test_parse_llama_cpp_performance_summary():
    stderr = """
llama_perf_context_print: prompt eval time = 120.50 ms / 100 tokens
llama_perf_context_print: eval time = 80.25 ms / 16 runs
"""

    assert _parse_latency(stderr, "") == (120.50, 80.25)


def test_parse_latency_rejects_missing_component():
    with pytest.raises(RuntimeError, match="did not contain both"):
        _parse_latency("prompt eval time = 2.0 ms", "")


def test_parse_llama_cpp_throughput_summary():
    assert _parse_throughput("", "[ Prompt: 400.0 t/s | Generation: 50.0 t/s ]") == (400.0, 50.0)


def test_clean_generation_removes_llama_cli_prompt_echo_and_banner():
    stdout = """banner\n> What is 2+2?\n2+2 equals 4.\n[ Prompt: 400.0 t/s | Generation: 50.0 t/s ]\nExiting...\n"""
    assert _clean_generation(stdout) == "2+2 equals 4."


def test_llama_cli_backend_records_real_components_and_output():
    completed = type(
        "Completed",
        (),
        {
            "returncode": 0,
            "stdout": "Paris\n",
            "stderr": "prompt eval time = 10.0 ms\neval time = 20.0 ms\n",
        },
    )()
    backend = LlamaCliInferenceBackend(
        cli_path="llama-cli.exe",
        model_path="model.gguf",
        max_new_tokens=8,
    )

    with patch("ladc.inference.llama_cli.subprocess.run", return_value=completed) as run:
        result = backend.generate("Question: capital?", tokenizer=WhitespaceTokenizer())

    assert result.output_text == "Paris"
    assert result.input_tokens == 2
    assert result.output_tokens == 1
    assert result.prefill_ms == 10.0
    assert result.decode_ms == 20.0
    assert result.metadata["timing_source"] == "llama.cpp millisecond performance summary"
    command = run.call_args.args[0]
    assert command[:3] == ["llama-cli.exe", "-m", "model.gguf"]
    assert "--no-display-prompt" in command
    assert "--single-turn" in command
    assert "--perf" in command
