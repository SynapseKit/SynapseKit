from __future__ import annotations

from collections.abc import AsyncGenerator

from .base import BaseLLM, LLMConfig


class CohereLLM(BaseLLM):
    """Cohere provider with async streaming."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import cohere
            except ImportError:
                raise ImportError("cohere required: pip install synapsekit[cohere]") from None
            self._client = cohere.AsyncClientV2(api_key=self.config.api_key)
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
        stream = client.chat_stream(
            model=self.config.model,
            messages=messages,
            temperature=kw.get("temperature", self.config.temperature),
            max_tokens=kw.get("max_tokens", self.config.max_tokens),
        )
        async for event in stream:
            if hasattr(event, "delta") and event.delta and event.delta.message:
                content = event.delta.message.content
                if content and content.text:
                    self._output_tokens += 1
                    yield content.text
