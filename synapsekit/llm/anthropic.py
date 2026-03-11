from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List

from .base import BaseLLM, LLMConfig


class AnthropicLLM(BaseLLM):
    """Anthropic Messages API with async streaming."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required: pip install synapsekit[anthropic]"
                )
            self._client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
        return self._client

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str, None]:
        client = self._get_client()
        async with client.messages.stream(
            model=self.config.model,
            system=self.config.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=kw.get("temperature", self.config.temperature),
            max_tokens=kw.get("max_tokens", self.config.max_tokens),
        ) as stream:
            async for text in stream.text_stream:
                yield text
            # Capture usage after stream completes
            message = await stream.get_final_message()
            self._input_tokens += message.usage.input_tokens or 0
            self._output_tokens += message.usage.output_tokens or 0

    async def stream_with_messages(
        self, messages: List[dict], **kw
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()
        # Filter out system messages — Anthropic takes system separately
        system = self.config.system_prompt
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
            else:
                user_messages.append(m)

        async with client.messages.stream(
            model=self.config.model,
            system=system,
            messages=user_messages,
            temperature=kw.get("temperature", self.config.temperature),
            max_tokens=kw.get("max_tokens", self.config.max_tokens),
        ) as stream:
            async for text in stream.text_stream:
                yield text
            message = await stream.get_final_message()
            self._input_tokens += message.usage.input_tokens or 0
            self._output_tokens += message.usage.output_tokens or 0

    async def call_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
    ) -> Dict[str, Any]:
        """Native tool use. Returns {"content": str|None, "tool_calls": list|None}."""
        client = self._get_client()
        system = self.config.system_prompt
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
            elif m.get("role") == "tool":
                # Convert OpenAI tool-result format to Anthropic
                user_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m["tool_call_id"],
                            "content": m["content"],
                        }
                    ],
                })
            else:
                user_messages.append(m)

        # Convert OpenAI tool schema → Anthropic tool schema
        anthropic_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]

        response = await client.messages.create(
            model=self.config.model,
            system=system,
            messages=user_messages,
            tools=anthropic_tools,
            max_tokens=self.config.max_tokens,
        )
        self._input_tokens += response.usage.input_tokens or 0
        self._output_tokens += response.usage.output_tokens or 0

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if tool_uses:
            return {
                "content": None,
                "tool_calls": [
                    {"id": tu.id, "name": tu.name, "arguments": tu.input}
                    for tu in tool_uses
                ],
            }
        text_blocks = [b for b in response.content if b.type == "text"]
        return {"content": text_blocks[0].text if text_blocks else "", "tool_calls": None}
