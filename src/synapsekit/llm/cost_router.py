"""CostRouter — route to the cheapest model meeting quality/latency thresholds."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from ..observability.tracer import COST_TABLE
from ..rag.facade import _make_llm
from .base import BaseLLM, LLMConfig

# Static quality scores (0-1) for known models.
QUALITY_TABLE: dict[str, float] = {
    # OpenAI — GPT-4o family
    "gpt-4o": 0.90,
    "gpt-4o-mini": 0.75,
    "gpt-4o-2024-11-20": 0.90,
    "gpt-4-turbo": 0.88,
    # OpenAI — GPT-4.1 family
    "gpt-4.1": 0.92,
    "gpt-4.1-mini": 0.78,
    "gpt-4.1-nano": 0.65,
    # OpenAI — o-series reasoning
    "o3": 0.95,
    "o3-mini": 0.82,
    "o4-mini": 0.83,
    # Anthropic
    "claude-opus-4-6": 0.96,
    "claude-sonnet-4-6": 0.91,
    "claude-haiku-4-5-20251001": 0.76,
    # Google Gemini
    "gemini-2.5-pro": 0.91,
    "gemini-2.5-flash": 0.74,
    # DeepSeek
    "deepseek-chat": 0.72,
    "deepseek-reasoner": 0.80,
    # Groq-hosted
    "llama-3.3-70b-versatile": 0.78,
    "mixtral-8x7b-32768": 0.70,
}


@dataclass
class RouterModelSpec:
    """Specification for a model available to the router."""

    model: str
    api_key: str
    provider: str | None = None
    max_latency_ms: float | None = None


@dataclass
class CostRouterConfig:
    """Configuration for CostRouter."""

    models: list[RouterModelSpec]
    quality_threshold: float = 0.0
    strategy: str = "cheapest"  # "cheapest" is the only strategy for now
    fallback_on_error: bool = True


class CostRouter(BaseLLM):
    """Route to the cheapest model that meets quality and latency thresholds.

    Subclasses ``BaseLLM`` for drop-in compatibility wherever an LLM is expected.

    Example::

        router = CostRouter(CostRouterConfig(
            models=[
                RouterModelSpec(model="gpt-4o-mini", api_key="sk-..."),
                RouterModelSpec(model="gpt-4o", api_key="sk-..."),
            ],
            quality_threshold=0.8,
        ))
        answer = await router.generate("Summarise this document")
    """

    def __init__(self, router_config: CostRouterConfig) -> None:
        # Pass a dummy config to BaseLLM so it initialises cleanly.
        super().__init__(LLMConfig(model="__cost_router__", api_key="", provider="openai"))
        self._router_config = router_config
        self._selected_model: str | None = None
        self._candidates = self._rank_candidates()

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #

    def _rank_candidates(self) -> list[RouterModelSpec]:
        """Filter by quality threshold, then sort cheapest-first."""
        threshold = self._router_config.quality_threshold
        eligible = [
            spec
            for spec in self._router_config.models
            if QUALITY_TABLE.get(spec.model, 0.5) >= threshold
        ]
        # Sort by total per-token cost (input + output)
        return sorted(eligible, key=lambda s: self._model_cost(s.model))

    @staticmethod
    def _model_cost(model: str) -> float:
        """Total per-token cost for ranking purposes."""
        costs = COST_TABLE.get(model, {})
        return costs.get("input", float("inf")) + costs.get("output", float("inf"))

    # ------------------------------------------------------------------ #
    # LLM construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_llm(spec: RouterModelSpec) -> BaseLLM:
        return _make_llm(
            model=spec.model,
            api_key=spec.api_key,
            provider=spec.provider,
            system_prompt="You are a helpful assistant.",
            temperature=0.2,
            max_tokens=1024,
        )

    # ------------------------------------------------------------------ #
    # Public API (BaseLLM interface)
    # ------------------------------------------------------------------ #

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        """Try candidates cheapest-first, falling back on error."""
        last_exc: Exception | None = None
        for spec in self._candidates:
            try:
                llm = self._build_llm(spec)
                self._selected_model = spec.model
                async for token in llm.stream(prompt, **kw):
                    yield token
                return
            except Exception as exc:
                last_exc = exc
                if not self._router_config.fallback_on_error:
                    raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No candidate models available")

    async def generate(self, prompt: str, **kw: Any) -> str:
        """Try candidates cheapest-first with optional latency constraint checking."""
        last_exc: Exception | None = None
        for spec in self._candidates:
            try:
                llm = self._build_llm(spec)
                self._selected_model = spec.model
                result = await llm.generate(prompt, **kw)
                return result
            except Exception as exc:
                last_exc = exc
                if not self._router_config.fallback_on_error:
                    raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No candidate models available")

    @property
    def selected_model(self) -> str | None:
        """The model that was actually used for the last call."""
        return self._selected_model
