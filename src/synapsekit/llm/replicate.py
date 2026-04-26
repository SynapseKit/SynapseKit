"""Replicate inference provider for SynapseKit."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseLLM, LLMConfig


class ReplicateLLM(BaseLLM):
    """Replicate inference provider.

    Install: ``pip install synapsekit[replicate]``  (requires ``replicate>=0.25``).
    """

    def __init__(
        self,
        model: str = "meta/llama-3-8b-instruct",
        api_key: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        resolved_key = api_key or os.environ.get("REPLICATE_API_TOKEN", "")
        config = LLMConfig(
            model=model,
            api_key=resolved_key,
            provider="replicate",
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        super().__init__(config)
        self._max_new_tokens = max_new_tokens
        self._top_p = top_p
        self._api_key = resolved_key

    def _get_replicate(self) -> Any:
        try:
            import replicate  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "replicate package required: pip install synapsekit[replicate]"
            ) from None
        if self._api_key:
            import replicate.client as _rc  # type: ignore[import-untyped]

            return _rc.Client(api_token=self._api_key)
        return replicate

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        client = self._get_replicate()
        input_payload = {
            "prompt": prompt,
            "max_new_tokens": kw.get("max_new_tokens", self._max_new_tokens),
            "temperature": kw.get("temperature", self.config.temperature),
            "top_p": kw.get("top_p", self._top_p),
        }
        output = await client.async_run(self.config.model, input=input_payload)
        # output may be a list of strings or an async iterator
        if hasattr(output, "__aiter__"):
            async for token in output:
                if token:
                    self._output_tokens += 1
                    yield str(token)
        else:
            tokens = list(output) if not isinstance(output, list) else output
            for token in tokens:
                if token:
                    self._output_tokens += 1
                    yield str(token)
