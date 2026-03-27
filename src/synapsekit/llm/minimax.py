from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig

_MINIMAX_BASE_URL = "https://api.minimax.chat/v1"


class MinimaxLLM(BaseLLM):
    """Minimax LLM provider via native REST API.

    Uses the ``/chat/completions`` endpoint with optional SSE streaming.
    Models include ``abab5.5-chat`` and other ``abab`` chat models.
    """

    def __init__(
        self,
        config: LLMConfig,
        group_id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(config)
        self._group_id = group_id
        self._base_url = (base_url or _MINIMAX_BASE_URL).rstrip("/")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._group_id:
                import os

                self._group_id = os.environ.get("MINIMAX_GROUP_ID")
            if not self._group_id:
                raise ValueError(
                    "group_id is required for MinimaxLLM. Pass group_id or set MINIMAX_GROUP_ID."
                )
            try:
                import httpx
            except ImportError:
                raise ImportError("httpx required: pip install synapsekit[minimax]") from None
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                timeout=120.0,
            )
        return self._client

    def _endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

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
        client = self._get_client()
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "tokens_to_generate": kw.get("max_tokens", self.config.max_tokens),
            "temperature": kw.get("temperature", self.config.temperature),
        }
        params = {"GroupId": self._group_id}
        async with client.stream("POST", self._endpoint(), params=params, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: ") :]
                else:
                    data = line
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = _extract_stream_text(chunk)
                if text:
                    self._output_tokens += 1
                    yield text


def _extract_stream_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "delta" in choice and isinstance(choice["delta"], dict):
        return str(choice["delta"].get("content", ""))
    if "messages" in choice and isinstance(choice["messages"], list):
        parts = [m.get("text", "") for m in choice["messages"] if m.get("text")]
        return "".join(parts)
    if "text" in choice:
        return str(choice.get("text", ""))
    return ""
