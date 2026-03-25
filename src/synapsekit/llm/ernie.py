from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig


class ErnieLLM(BaseLLM):
    """Baidu ERNIE Bot LLM provider with async streaming.

    Supports models like ``ernie-3.5``, ``ernie-4.0``, ``ernie-lite``,
    ``ernie-speed``, ``ernie-tiny-8k``.
    """

    def __init__(
        self,
        config: LLMConfig,
        api_type: str = "aistudio",
    ) -> None:
        super().__init__(config)
        self._api_type = api_type
        self._configured = False

    def _configure(self) -> None:
        """Configure erniebot module settings."""
        if self._configured:
            return
        try:
            import erniebot
        except ImportError:
            raise ImportError("erniebot package required: pip install synapsekit[ernie]") from None
        erniebot.api_type = self._api_type
        erniebot.access_token = self.config.api_key
        self._configured = True

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        messages = [
            {"role": "user", "content": prompt},
        ]
        async for token in self.stream_with_messages(messages, **kw):
            yield token

    async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:
        self._configure()
        import erniebot

        # Inject system prompt if provided
        system = self.config.system_prompt if self.config.system_prompt else None

        # When stream=True, acreate returns an AsyncIterator directly
        stream = erniebot.ChatCompletion.acreate(
            model=self.config.model,
            messages=messages,
            system=system,
            temperature=kw.get("temperature", self.config.temperature),
            max_output_tokens=kw.get("max_tokens", self.config.max_tokens),
            stream=True,
        )
        async for chunk in stream:
            result = chunk.get_result()
            if result:
                self._output_tokens += 1
                yield result

    async def _call_with_tools_impl(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Native function-calling via ERNIE's functions parameter."""
        self._configure()
        import erniebot

        # Convert OpenAI-style tools to ERNIE functions format
        functions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                functions.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )

        system = self.config.system_prompt if self.config.system_prompt else None

        response = await erniebot.ChatCompletion.acreate(
            model=self.config.model,
            messages=messages,
            functions=functions,
            system=system,
            stream=False,
        )

        # Check if response contains function call
        if hasattr(response, "function_call") and response.function_call:
            func_call = response.function_call
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "name": func_call.get("name", ""),
                        "arguments": json.loads(func_call.get("arguments", "{}")),
                    }
                ],
            }

        return {"content": response.get_result(), "tool_calls": None}
