"""Tokenizer adapters used by workload generation and runtime evaluation."""

from __future__ import annotations

import re


class HuggingFaceTokenizer:
    """Adapter around a Transformers tokenizer for native token accounting.

    The dependency is imported only when this adapter is constructed, so the
    core tests remain usable without installing Transformers. Special tokens
    are disabled to make token counts match the plain prompt passed to the
    inference backend; the backend configuration must use the same convention.
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        local_files_only: bool = True,
        **kwargs,
    ) -> "HuggingFaceTokenizer":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "HuggingFaceTokenizer requires Transformers; install the native tokenizer extra"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            use_fast=True,
            **kwargs,
        )
        return cls(tokenizer)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return self._tokenizer.decode(
            token_ids[:max_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def count_chat_tokens(self, user_text: str, *, system_prompt: str | None = None) -> int:
        """Count the actual model input after applying the tokenizer chat template."""

        return len(self._chat_token_ids(user_text, system_prompt=system_prompt))

    def format_chat_prompt(self, user_text: str, *, system_prompt: str | None = None) -> str:
        """Render the chat template as text for runtimes accepting tokenized prompts."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _chat_token_ids(self, user_text: str, *, system_prompt: str | None = None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )


class WhitespaceTokenizer:
    """Deterministic fallback tokenizer for pipeline tests.

    It is deliberately not used for final model claims. Final experiments must
    inject the target model's native tokenizer through the Tokenizer protocol.
    """

    _token_re = re.compile(r"\S+")

    def count_tokens(self, text: str) -> int:
        return len(self._token_re.findall(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokens = self._token_re.findall(text)
        return " ".join(tokens[:max_tokens])
