from __future__ import annotations

from collections.abc import AsyncGenerator

from .base import BaseLLM, LLMConfig


class OllamaLLM(BaseLLM):
    """Ollama local model provider with async streaming."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from ollama import AsyncClient
            except ImportError:
                raise ImportError("ollama required: pip install synapsekit[ollama]") from None
            self._client = AsyncClient()
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
        async for chunk in client.chat(
            model=self.config.model,
            messages=messages,
            stream=True,
            options={
                "temperature": kw.get("temperature", self.config.temperature),
                "num_predict": kw.get("max_tokens", self.config.max_tokens),
            },
        ):
            content = chunk["message"]["content"]
            if content:
                self._output_tokens += 1
                yield content
