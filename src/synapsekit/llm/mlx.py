"""Apple Silicon local LLM provider using mlx-lm."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig, _messages_to_prompt


class MLXLLM(BaseLLM):
    """Local provider for MLX models on Apple Silicon.

    ``mlx-lm`` exposes a few generation APIs across versions, so this wrapper
    supports both simple string generation and iterable token streams.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        adapter_path: str | None = None,
        tokenizer_config: dict[str, Any] | None = None,
        **load_kwargs: Any,
    ) -> None:
        super().__init__(config)
        self._adapter_path = adapter_path
        self._tokenizer_config = tokenizer_config or {}
        self._load_kwargs = load_kwargs
        self._model: Any = None
        self._tokenizer: Any = None
        self._generate: Any = None
        self._stream_generate: Any = None

    def _load_backend(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            from mlx_lm import generate, load, stream_generate
        except ImportError:
            raise ImportError("mlx-lm required: pip install synapsekit[mlx]") from None

        kwargs = dict(self._load_kwargs)
        if self._adapter_path is not None:
            kwargs["adapter_path"] = self._adapter_path
        if self._tokenizer_config:
            kwargs["tokenizer_config"] = self._tokenizer_config

        self._model, self._tokenizer = load(self.config.model, **kwargs)
        self._generate = generate
        self._stream_generate = stream_generate
        return self._model, self._tokenizer

    @staticmethod
    def _extract_text(token: Any) -> str:
        if isinstance(token, str):
            return token
        text = getattr(token, "text", None)
        if text is not None:
            return str(text)
        return str(token)

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        async for token in self.stream_with_messages(messages, **kw):
            yield token

    async def stream_with_messages(
        self,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> AsyncGenerator[str]:
        model, tokenizer = self._load_backend()
        prompt = _messages_to_prompt(messages)
        max_tokens = int(kw.get("max_tokens", self.config.max_tokens))
        temperature = float(kw.get("temperature", self.config.temperature))

        if self._stream_generate is None:
            raise RuntimeError("MLX stream generator was not initialized")

        def _collect() -> list[str]:
            tokens: list[str] = []
            for token in self._stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
            ):
                text = self._extract_text(token)
                if text:
                    tokens.append(text)
            return tokens

        loop = asyncio.get_running_loop()
        tokens = await loop.run_in_executor(None, _collect)
        for token in tokens:
            self._output_tokens += 1
            yield token

    async def generate(self, prompt: str, **kw: Any) -> str:  # type: ignore[override]
        model, tokenizer = self._load_backend()
        max_tokens = int(kw.get("max_tokens", self.config.max_tokens))
        temperature = float(kw.get("temperature", self.config.temperature))

        if self._generate is None:
            return await super().generate(prompt, **kw)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
            ),
        )
        text = self._extract_text(result)
        self._output_tokens += max(1, len(text.split()))
        return text
