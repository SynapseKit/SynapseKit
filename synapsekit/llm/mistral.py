from __future__ import annotations

from typing import AsyncGenerator, List

from .base import BaseLLM, LLMConfig


class MistralLLM(BaseLLM):
    """Mistral AI provider with async streaming."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from mistralai import Mistral
            except ImportError:
                raise ImportError("mistralai required: pip install synapsekit[mistral]")
            self._client = Mistral(api_key=self.config.api_key)
        return self._client

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        async for token in self.stream_with_messages(messages, **kw):
            yield token

    async def stream_with_messages(
        self, messages: List[dict], **kw
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()
        async for chunk in client.chat.stream_async(
            model=self.config.model,
            messages=messages,
            temperature=kw.get("temperature", self.config.temperature),
            max_tokens=kw.get("max_tokens", self.config.max_tokens),
        ):
            delta = chunk.data.choices[0].delta.content
            if delta:
                self._output_tokens += 1
                yield delta
