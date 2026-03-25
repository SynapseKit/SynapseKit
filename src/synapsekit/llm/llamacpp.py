from __future__ import annotations

from collections.abc import AsyncGenerator

from .base import BaseLLM, LLMConfig


class LlamaCppLLM(BaseLLM):
    """Llama.cpp local GGUF provider with async streaming via llama-cpp-python."""

    def __init__(self, config: LLMConfig, model_path: str | None = None) -> None:
        super().__init__(config)
        self._model_path = model_path or config.model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python package required: pip install synapsekit[llamacpp]"
                ) from None
            if not self._model_path:
                raise ValueError("model_path is required for llama.cpp")
            self._client = Llama(model_path=self._model_path)
        return self._client

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        async for token in self.stream_with_messages(messages, **kw):
            yield token

    async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:
        client = self._get_client()
        stream = client.create_chat_completion(
            messages=messages,
            temperature=kw.get("temperature", self.config.temperature),
            top_p=kw.get("top_p", 1.0),
            max_tokens=kw.get("max_tokens", self.config.max_tokens),
            stream=True,
        )
        for chunk in stream:
            choice = (chunk or {}).get("choices", [{}])[0]
            delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
            content = delta.get("content") if isinstance(delta, dict) else None
            if content:
                self._output_tokens += 1
                yield content
