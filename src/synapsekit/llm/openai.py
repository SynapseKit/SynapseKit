from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List

from .base import BaseLLM, LLMConfig


class OpenAILLM(BaseLLM):
    """OpenAI chat completions with async streaming."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install synapsekit[openai]"
                )
            self._client = AsyncOpenAI(api_key=self.config.api_key)
        return self._client

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        client = self._get_client()
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt},
        ]
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

    async def stream_with_messages(
        self, messages: List[dict], **kw
    ) -> AsyncGenerator[str, None]:
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

    async def call_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
    ) -> Dict[str, Any]:
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
