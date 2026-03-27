from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig


class AlephAlphaLLM(BaseLLM):
    """Aleph Alpha LLM provider using aleph-alpha-client.

    Supports Luminous and Pharia models (e.g., ``luminous-supreme``,
    ``pharia-1-llm-7b-control``).
    """

    def __init__(
        self,
        config: LLMConfig,
        host: str | None = None,
    ) -> None:
        super().__init__(config)
        self._host = host
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from aleph_alpha_client import AsyncClient
            except ImportError:
                raise ImportError(
                    "aleph-alpha-client required: pip install synapsekit[aleph-alpha]"
                ) from None
            import os

            self._client = AsyncClient(
                token=self.config.api_key,
                host=self._host or os.environ.get("AA_API_URL", "https://api.aleph-alpha.com"),
            )
        return self._client

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
        from aleph_alpha_client import CompletionRequest, Prompt

        client = self._get_client()
        # Aleph Alpha uses completion API with a prompt
        # For chat-style messages, we convert to text prompt
        prompt_text = self._messages_to_prompt(messages)
        request = CompletionRequest(
            prompt=Prompt.from_text(prompt_text),
            maximum_tokens=kw.get("max_tokens", self.config.max_tokens),
            temperature=kw.get("temperature", self.config.temperature),
        )
        response_stream = client.complete_with_streaming(request, model=self.config.model)
        async for stream_item in response_stream:
            if stream_item.completion:
                self._output_tokens += 1
                yield stream_item.completion

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert OpenAI-style messages to a text prompt."""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        return "\n".join(parts)
