"""Local Transformers inference backend with separate prefill/decode timing."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ladc.interfaces import Tokenizer
from ladc.records import InferenceRecord


def _synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TransformersInferenceBackend:
    """Greedy local generation for runtime smoke tests.

    The first forward pass is reported as prefill latency. Subsequent cached
    one-token forward passes are reported as decode latency. The class is
    intentionally separate from the llama.cpp backend because runtimes and
    latency semantics must not be mixed in one result table.
    """

    name = "transformers"

    def __init__(
        self,
        *,
        model,
        tokenizer,
        device: str = "cuda",
        max_new_tokens: int = 64,
        model_id: str = "unknown",
    ) -> None:
        import torch

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        tokenizer_path: str | None = None,
        device: str = "cuda",
        torch_dtype: Any = "auto",
        max_new_tokens: int = 64,
        local_files_only: bool = True,
    ) -> "TransformersInferenceBackend":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("TransformersInferenceBackend requires torch and transformers") from exc

        tokenizer_source = tokenizer_path or model_path
        hf_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=local_files_only,
            use_fast=True,
        )
        dtype = torch_dtype
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            dtype=dtype,
        )
        model.to(device)
        return cls(
            model=model,
            tokenizer=hf_tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            model_id=model_path,
        )

    def generate(self, prompt: str, *, tokenizer: Tokenizer) -> InferenceRecord:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        _synchronize(self.device)
        prefill_start = perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        _synchronize(self.device)
        prefill_ms = (perf_counter() - prefill_start) * 1000.0

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated: list[Any] = [next_token]
        eos_token_id = self.tokenizer.eos_token_id

        _synchronize(self.device)
        decode_start = perf_counter()
        with torch.inference_mode():
            for _ in range(self.max_new_tokens - 1):
                if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                    break
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=self.device, dtype=attention_mask.dtype)],
                        dim=1,
                    )
                step_outputs = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = step_outputs.past_key_values
                next_token = step_outputs.logits[:, -1:, :].argmax(dim=-1)
                generated.append(next_token)
        _synchronize(self.device)
        decode_ms = (perf_counter() - decode_start) * 1000.0

        generated_ids = torch.cat(generated, dim=1)[0].tolist()
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        input_tokens = tokenizer.count_tokens(prompt)
        return InferenceRecord(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=tokenizer.count_tokens(output_text),
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            metadata={
                "backend": self.name,
                "model_id": self.model_id,
                "device": str(self.device),
                "timing_source": "torch synchronized wall clock",
                "generated_token_count": len(generated_ids),
            },
        )
