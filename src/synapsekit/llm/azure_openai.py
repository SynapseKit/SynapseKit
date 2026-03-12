from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI chat completions with async streaming.

    Requires ``api_key``, and the following config fields set via
    ``LLMConfig`` or constructor kwargs:

    - ``model``: The Azure deployment name (e.g. ``"gpt-4o"``).
    - ``api_key``: Azure OpenAI API key.
    - ``azure_endpoint``: Your Azure resource URL
      (e.g. ``"https://myresource.openai.azure.com"``).
    - ``api_version``: Azure API version (default ``"2024-06-01"``).
    """

    def __init__(
        self,
        config: LLMConfig,
        azure_endpoint: str | None = None,
        api_version: str = "2024-06-01",
    ) -> None:
        super().__init__(config)
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install synapsekit[openai]"
                ) from None
            if not self._azure_endpoint:
                raise ValueError(
                    "azure_endpoint is required for AzureOpenAILLM. "
                    "Pass it to the constructor: AzureOpenAILLM(config, azure_endpoint='...')"
                )
            self._client = AsyncAzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        return self._client

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
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
