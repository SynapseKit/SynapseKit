from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig

_DEFAULT_BASE_URL = "http://localhost:1234/v1"


class LMStudioLLM(BaseLLM):
    """LM Studio local model provider via its OpenAI-compatible API.

    Connects to a running LM Studio server (default: http://localhost:1234/v1).
    Requires the ``openai`` package (``pip install synapsekit[lmstudio]``).

    Usage::

        from synapsekit.llm.base import LLMConfig
        from synapsekit.llm.lmstudio import LMStudioLLM

        llm = LMStudioLLM(LLMConfig(model="llama-3-8b-instruct", provider="lmstudio"))
        response = await llm.generate("Hello!")

    Custom server URL::

        config = LLMConfig(
            model="mistral-7b-instruct",
            provider="lmstudio",
            base_url="http://192.168.1.10:1234/v1",
        )
        llm = LMStudioLLM(config)
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None
        self._base_url: str = getattr(config, "base_url", None) or _DEFAULT_BASE_URL

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install synapsekit[lmstudio]"
                ) from None
            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self.config.api_key or "lm-studio",
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
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage:
                self._input_tokens += chunk.usage.prompt_tokens or 0
                self._output_tokens += chunk.usage.completion_tokens or 0
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def _call_with_tools_impl(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Native function-calling. Returns {"content": str|None, "tool_calls": list|None}."""
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message
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
