"""CloudflareLLM — Cloudflare Workers AI via native REST API.

Based on the implementation by @adaumsilva (PR #320).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig

_CF_BASE = "https://api.cloudflare.com/client/v4/accounts"


class CloudflareLLM(BaseLLM):
    """Cloudflare Workers AI LLM provider via httpx REST API.

    Uses the native ``/ai/run/`` endpoint with SSE streaming.
    Models include ``@cf/meta/llama-3.1-8b-instruct``,
    ``@cf/mistral/mistral-7b-instruct-v0.1``, etc.
    """

    def __init__(
        self,
        config: LLMConfig,
        account_id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(config)
        self._account_id = account_id
        self._base_url = (base_url or _CF_BASE).rstrip("/")
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._account_id:
                raise ValueError(
                    "account_id is required for CloudflareLLM. "
                    "Pass account_id to the constructor: CloudflareLLM(config, account_id='...')"
                )

            try:
                import httpx
            except ImportError:
                raise ImportError("httpx required: pip install synapsekit[cloudflare]") from None

            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    def _endpoint(self) -> str:
        return f"{self._base_url}/{self._account_id}/ai/run/{self.config.model}"

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
            "messages": messages,
            "stream": True,
            "max_tokens": kw.get("max_tokens", self.config.max_tokens),
            "temperature": kw.get("temperature", self.config.temperature),
        }
        async with client.stream("POST", self._endpoint(), json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = chunk.get("response", "")
                if text:
                    self._output_tokens += 1
                    yield text
