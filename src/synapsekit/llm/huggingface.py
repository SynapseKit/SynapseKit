from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig

try:
    from huggingface_hub import AsyncInferenceClient

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class HuggingFaceLLM(BaseLLM):
    """LLM provider for Hugging Face Inference API / Endpoints.

    Uses `huggingface-hub`'s `AsyncInferenceClient`.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "huggingface-hub is required for HuggingFaceLLM. "
                "Install with `pip install synapsekit[huggingface]`."
            )

        api_key = self.config.api_key or os.getenv("HUGGINGFACE_API_KEY")

        # If config.model is a full URL, it's an Inference Endpoint
        # Otherwise, it's treated as a model ID for serverless API
        kwargs: dict[str, Any] = {"token": api_key}
        if self.config.model.startswith("http"):
            kwargs["model"] = self.config.model
        else:
            kwargs["model"] = self.config.model or "mistralai/Mixtral-8x7B-Instruct-v0.1"

        self.client = AsyncInferenceClient(**kwargs)

    async def _generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        params = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens or 1024),
            "temperature": kwargs.get("temperature", self.config.temperature or 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": False,
        }

        # Optional param adjustments
        if "stop" in kwargs:
            params["stop_sequences"] = kwargs["stop"]

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        response = await self.client.text_generation(prompt=full_prompt, **params)
        return str(response)

    async def stream(
        self, prompt: str, system: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        params = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens or 1024),
            "temperature": kwargs.get("temperature", self.config.temperature or 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "stream": True,
        }

        if "stop" in kwargs:
            params["stop_sequences"] = kwargs["stop"]

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        async for chunk in await self.client.text_generation(prompt=full_prompt, **params):
            if chunk:
                yield chunk
