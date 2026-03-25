from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig


class DatabricksLLM(BaseLLM):
    """Databricks Foundation Model APIs with OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: LLMConfig,
        workspace_url: str | None = None,
    ) -> None:
        super().__init__(config)
        self._client = None
        self._workspace_url = workspace_url

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install synapsekit[openai]"
                ) from None
            import os

            workspace = self._workspace_url or os.environ.get("DATABRICKS_HOST", "")
            if not workspace:
                raise ValueError(
                    "Databricks workspace URL required. Set DATABRICKS_HOST or pass workspace_url."
                )
            base_url = f"{workspace.rstrip('/')}/serving-endpoints"
            self._client = AsyncOpenAI(api_key=self.config.api_key, base_url=base_url)
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
                yield chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage:
                self._input_tokens += chunk.usage.prompt_tokens or 0
                self._output_tokens += chunk.usage.completion_tokens or 0

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
