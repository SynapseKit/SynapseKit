from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig


class CloudflareLLM(BaseLLM):
    """Cloudflare Workers AI LLM provider with async streaming.

    Uses the OpenAI-compatible API. Requires an ``account_id`` and an
    API token as ``api_key``.

    Models include ``@hf/noushermes-13b-warm``, ``@cf/meta/llama-3-8b-instruct``, etc.
    """

    def __init__(
        self,
        config: LLMConfig,
        account_id: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(config)
        self._account_id = account_id
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install synapsekit[openai]"
                ) from None

            if not self._base_url:
                if not self._account_id:
                    raise ValueError(
                        "Either account_id or base_url is required for CloudflareLLM. "
                        "Pass account_id to the constructor: CloudflareLLM(config, account_id='...')"
                    )
                self._base_url = (
                    f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}/ai/v1"
                )

            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self._base_url,
            )
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
        stream = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kw.get("temperature", self.config.temperature),
            max_tokens=kw.get("max_tokens", self.config.max_tokens),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                self._output_tokens += 1
                yield chunk.choices[0].delta.content

    async def _call_with_tools_impl(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        if hasattr(response, "usage") and response.usage:
            self._input_tokens += response.usage.prompt_tokens or 0
            self._output_tokens += response.usage.completion_tokens or 0

        if msg.tool_calls:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                    for tc in msg.tool_calls
                ],
            }
        return {"content": msg.content, "tool_calls": None}
