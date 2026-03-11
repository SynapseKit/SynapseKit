from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig, _messages_to_prompt


class GeminiLLM(BaseLLM):
    """Google Gemini provider with async streaming."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai required: pip install synapsekit[gemini]"
                ) from None
            genai.configure(api_key=self.config.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.config.model,
                system_instruction=self.config.system_prompt,
            )
        return self._model

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        model = self._get_model()
        async for chunk in await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": kw.get("temperature", self.config.temperature),
                "max_output_tokens": kw.get("max_tokens", self.config.max_tokens),
            },
            stream=True,
        ):
            if chunk.text:
                self._output_tokens += 1
                yield chunk.text

    async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:
        prompt = _messages_to_prompt(messages)
        async for token in self.stream(prompt, **kw):
            yield token

    async def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Native function-calling. Returns {"content": str|None, "tool_calls": list|None}."""
        import google.generativeai as genai

        model = self._get_model()

        # Convert OpenAI tool schema → Gemini function declarations
        func_decls = []
        for t in tools:
            fn = t["function"]
            func_decls.append(
                genai.protos.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=self._convert_params(fn.get("parameters", {})),
                )
            )
        gemini_tools = [genai.protos.Tool(function_declarations=func_decls)]

        # Build prompt from messages
        prompt = _messages_to_prompt(messages)

        response = await model.generate_content_async(
            prompt,
            tools=gemini_tools,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
        )

        # Parse response parts for function calls vs text
        tool_calls = []
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args) if part.function_call.args else {},
                    }
                )
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if tool_calls:
            return {"content": None, "tool_calls": tool_calls}
        return {"content": "".join(text_parts) if text_parts else "", "tool_calls": None}

    @staticmethod
    def _convert_params(params: dict) -> dict:
        """Convert JSON Schema parameters to Gemini-compatible format."""
        if not params:
            return {}
        # Gemini accepts a subset of JSON Schema
        result: dict[str, Any] = {"type": params.get("type", "object").upper()}
        if "properties" in params:
            result["properties"] = {}
            for name, prop in params["properties"].items():
                result["properties"][name] = {
                    "type": prop.get("type", "string").upper(),
                    "description": prop.get("description", ""),
                }
        if "required" in params:
            result["required"] = params["required"]
        return result
