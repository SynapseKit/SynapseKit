"""Reka OpenAI-compatible LLM provider."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig

_REKA_BASE_URL = "https://api.reka.ai/v1"


class RekaLLM(BaseLLM):
    """Reka chat completions over its OpenAI-compatible API."""

    def __init__(self, config: LLMConfig, base_url: str | None = None) -> None:
        super().__init__(config)
        self._base_url = base_url or _REKA_BASE_URL
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required: pip install synapsekit[reka]") from None
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self._base_url,
                **({"timeout": self.config.timeout} if self.config.timeout is not None else {}),
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
        stream = await self._get_client().chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kw.get("temperature", self.config.temperature),
            max_tokens=kw.get("max_tokens", self.config.max_tokens),
            top_p=kw.get("top_p", self.config.top_p),
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage:
                self._input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self._output_tokens += getattr(usage, "completion_tokens", 0) or 0
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _call_with_tools_impl(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        response = await self._get_client().chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        usage = getattr(response, "usage", None)
        if usage:
            self._input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._output_tokens += getattr(usage, "completion_tokens", 0) or 0

        message = response.choices[0].message
        if message.tool_calls:
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": (
                            json.loads(tool_call.function.arguments)
                            if isinstance(tool_call.function.arguments, str)
                            else tool_call.function.arguments
                        ),
                    }
                    for tool_call in message.tool_calls
                ],
            }
        return {"content": message.content, "tool_calls": None}
