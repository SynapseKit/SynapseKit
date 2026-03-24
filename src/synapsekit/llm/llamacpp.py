from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig


class LlamaCppLLM(BaseLLM):
    """Local LLM provider using llama-cpp-python with GGUF models.

    Runs models entirely on-device — no API key required.
    Models: any GGUF file, e.g. ``llama-3.1-8b-instruct.Q4_K_M.gguf``.

    Example::

        llm = LlamaCppLLM(
            config=LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp"),
            model_path="/models/llama-3.1-8b-instruct.Q4_K_M.gguf",
            n_gpu_layers=35,
        )
        result = await llm.generate("What is the capital of France?")
    """

    def __init__(
        self,
        config: LLMConfig,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        top_p: float = 0.95,
        **llama_kwargs: Any,
    ) -> None:
        super().__init__(config)
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._top_p = top_p
        self._llama_kwargs = llama_kwargs
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python required: pip install synapsekit[llamacpp]"
                ) from None
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                **self._llama_kwargs,
            )
        return self._model

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        async for token in self.stream_with_messages(messages, **kw):
            yield token

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str]:
        model = self._get_model()
        temperature = kw.get("temperature", self.config.temperature)
        max_tokens = kw.get("max_tokens", self.config.max_tokens)
        top_p = kw.get("top_p", self._top_p)

        chunks = await asyncio.to_thread(
            lambda: list(
                model.create_chat_completion(
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            )
        )
        for chunk in chunks:
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                self._output_tokens += 1
                yield content
