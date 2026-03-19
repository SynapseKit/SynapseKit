from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig, _messages_to_prompt


class VertexAILLM(BaseLLM):
    """Google Vertex AI provider with async streaming.

    Uses Application Default Credentials (ADC) for authentication.
    The ``config.api_key`` field is used as the GCP project ID.

    Requires ``google-cloud-aiplatform`` package::

        pip install synapsekit[vertex]
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._model = None

    def _get_model(self):
        """Lazy initialization of the Vertex AI model."""
        if self._model is None:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
            except ImportError:
                raise ImportError(
                    "google-cloud-aiplatform required: pip install synapsekit[vertex]"
                ) from None
            vertexai.init(project=self.config.api_key)
            self._model = GenerativeModel(
                model_name=self.config.model,
                system_instruction=self.config.system_prompt,
            )
        return self._model

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        """Core streaming implementation."""
        model = self._get_model()
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": kw.get("temperature", self.config.temperature),
                "max_output_tokens": kw.get("max_tokens", self.config.max_tokens),
            },
            stream=True,
        )
        async for chunk in response:
            if chunk.text:
                self._output_tokens += 1
                yield chunk.text

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str]:
        """Stream from message list."""
        prompt = _messages_to_prompt(messages)
        async for token in self.stream(prompt, **kw):
            yield token

    async def _call_with_tools_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Native function-calling via Vertex AI."""
        from vertexai.generative_models import Tool as VertexTool

        model = self._get_model()

        # Convert OpenAI tool schema to Vertex function declarations
        func_decls = []
        for t in tools:
            fn = t["function"]
            func_decls.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": self._convert_params(fn.get("parameters", {})),
            })

        vertex_tools = [VertexTool.from_dict({"function_declarations": func_decls})]
        prompt = _messages_to_prompt(messages)

        response = await model.generate_content_async(
            prompt,
            tools=vertex_tools,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
        )

        tool_calls = []
        text_parts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args)
                    if part.function_call.args
                    else {},
                })
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if tool_calls:
            return {"content": None, "tool_calls": tool_calls}
        return {"content": "".join(text_parts) if text_parts else "", "tool_calls": None}

    @staticmethod
    def _convert_params(params: dict) -> dict:
        """Convert JSON Schema parameters to Vertex AI-compatible format."""
        if not params:
            return {}
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
