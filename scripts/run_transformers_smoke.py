"""Run one real local Transformers request and print latency components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ladc.inference import TransformersInferenceBackend
from ladc.tokenizers import HuggingFaceTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--prompt",
        default="Answer with one word. Question: What is 2+2? Answer:",
    )
    args = parser.parse_args()

    tokenizer = HuggingFaceTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    backend = TransformersInferenceBackend.from_pretrained(
        args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        torch_dtype="float16" if args.device.startswith("cuda") else "float32",
        max_new_tokens=args.max_new_tokens,
        local_files_only=True,
    )
    result = backend.generate(args.prompt, tokenizer=tokenizer)
    print(
        json.dumps(
            {
                "output": result.output_text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "prefill_ms": result.prefill_ms,
                "decode_ms": result.decode_ms,
                "end_to_end_ms": result.prefill_ms + result.decode_ms,
                "metadata": result.metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
