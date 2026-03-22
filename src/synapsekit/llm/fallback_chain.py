"""FallbackChain — try models in order, cascade on error or low confidence."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from .base import BaseLLM, LLMConfig


@dataclass
class FallbackChainConfig:
    """Configuration for FallbackChain."""

    models: list[BaseLLM]
    min_response_length: int = 1


class FallbackChain(BaseLLM):
    """Try models in order, escalating on error or short response.

    Takes pre-constructed ``BaseLLM`` instances — callers configure each independently.

    Example::

        chain = FallbackChain(FallbackChainConfig(
            models=[cheap_llm, expensive_llm],
            min_response_length=10,
        ))
        answer = await chain.generate("Explain quantum computing")
    """

    def __init__(self, chain_config: FallbackChainConfig) -> None:
        super().__init__(LLMConfig(model="__fallback_chain__", api_key="", provider="openai"))
        self._chain_config = chain_config
        self._used_model: BaseLLM | None = None

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        """Stream from first model that produces an adequate response."""
        last_exc: Exception | None = None
        min_len = self._chain_config.min_response_length

        for llm in self._chain_config.models:
            try:
                # Buffer the response to check min_response_length
                tokens: list[str] = []
                async for token in llm.stream(prompt, **kw):
                    tokens.append(token)
                full = "".join(tokens)
                if len(full.strip()) < min_len:
                    continue
                self._used_model = llm
                for token in tokens:
                    yield token
                return
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All models in fallback chain produced inadequate responses")

    async def generate(self, prompt: str, **kw: Any) -> str:
        """Generate from first model that produces an adequate response."""
        last_exc: Exception | None = None
        min_len = self._chain_config.min_response_length

        for llm in self._chain_config.models:
            try:
                result = await llm.generate(prompt, **kw)
                if len(result.strip()) < min_len:
                    continue
                self._used_model = llm
                return result
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All models in fallback chain produced inadequate responses")

    @property
    def used_model(self) -> BaseLLM | None:
        """The model that was actually used for the last call."""
        return self._used_model
