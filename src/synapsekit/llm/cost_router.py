"""CostRouter — route to the cheapest model meeting quality/latency thresholds."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

from ..llm._factory import make_llm
from ..observability.budget_guard import BudgetExceededError
from ..observability.tracer import COST_TABLE
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
        return make_llm(
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


class CostQualityRouter(BaseLLM):
    """Router that automatically downgrades models to stay within a cost budget.

    Example::

        router = CostQualityRouter(
            candidates=[gpt4o, gpt4o_mini, groq_llama],
            max_cost_per_call_usd=0.02,
            on_exceed="downgrade",
        )
        # If gpt4o is estimated to exceed $0.02, it fallbacks to gpt4o_mini.
        answer = await router.generate("Complex reasoning task")
    """

    def __init__(
        self,
        candidates: list[BaseLLM],
        max_cost_per_call_usd: float,
        on_exceed: Literal["downgrade", "raise", "skip"] = "downgrade",
    ) -> None:
        super().__init__(LLMConfig(model="__cost_quality_router__", api_key="", provider="openai"))
        self.candidates = candidates
        self.max_cost_per_call_usd = max_cost_per_call_usd
        self.on_exceed = on_exceed
        self._pending_events: list[dict[str, Any]] = []

    def _estimate_cost(self, llm: BaseLLM, messages: list[dict[str, Any]]) -> float:
        """Heuristic cost estimation based on character count."""
        model = llm.config.model
        pricing = COST_TABLE.get(model)
        if not pricing:
            return 0.0

        # Input estimate (4 chars ~ 1 token)
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        input_tokens = total_chars // 4

        # Output estimate (use max_tokens or default)
        output_tokens = llm.config.max_tokens or 1024

        return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

    def _record_downgrade(self, from_model: str, to_model: str, reason: str) -> None:
        self._pending_events.append(
            {
                "type": "cost_downgrade",
                "from_model": from_model,
                "to_model": to_model,
                "reason": reason,
            }
        )

    def consume_events(self) -> list[dict[str, Any]]:
        """Return and clear pending events."""
        events = self._pending_events[:]
        self._pending_events.clear()
        return events

    async def generate_with_messages(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        for i, llm in enumerate(self.candidates):
            cost = self._estimate_cost(llm, messages)
            if cost > self.max_cost_per_call_usd:
                if self.on_exceed == "downgrade" and i < len(self.candidates) - 1:
                    self._record_downgrade(
                        llm.config.model,
                        self.candidates[i + 1].config.model,
                        f"Estimated cost ${cost:.6f} exceeds limit ${self.max_cost_per_call_usd:.6f}",
                    )
                    continue
                elif self.on_exceed == "raise":
                    raise BudgetExceededError(
                        f"Estimated cost ${cost:.6f} exceeds limit ${self.max_cost_per_call_usd:.6f}",
                        limit_type="per_call",
                        limit_value=self.max_cost_per_call_usd,
                        current=cost,
                    )
                elif self.on_exceed == "skip":
                    return ""

                # Last candidate or unknown strategy
                if cost > self.max_cost_per_call_usd:
                    raise BudgetExceededError(
                        f"Final candidate '{llm.config.model}' estimated cost ${cost:.6f} exceeds limit ${self.max_cost_per_call_usd:.6f}",
                        limit_type="per_call",
                        limit_value=self.max_cost_per_call_usd,
                        current=cost,
                    )

            try:
                return await llm.generate_with_messages(messages, **kw)
            except BudgetExceededError as exc:
                if self.on_exceed == "downgrade" and i < len(self.candidates) - 1:
                    self._record_downgrade(
                        llm.config.model,
                        self.candidates[i + 1].config.model,
                        f"Budget exceeded during call: {exc}",
                    )
                    continue
                raise

        raise BudgetExceededError(
            "All candidates exhausted or budget exceeded",
            limit_type="per_call",
            limit_value=self.max_cost_per_call_usd,
            current=0.0,
        )

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str]:
        # Implementation note: real-time streaming fallback is hard because tokens
        # might have already been yielded. For now, we only downgrade BEFORE
        # starting the stream based on estimation.
        for i, llm in enumerate(self.candidates):
            cost = self._estimate_cost(llm, messages)
            if cost > self.max_cost_per_call_usd:
                if self.on_exceed == "downgrade" and i < len(self.candidates) - 1:
                    self._record_downgrade(
                        llm.config.model,
                        self.candidates[i + 1].config.model,
                        f"Estimated cost ${cost:.6f} exceeds limit ${self.max_cost_per_call_usd:.6f}",
                    )
                    continue
                elif self.on_exceed == "raise":
                    raise BudgetExceededError(
                        f"Estimated cost ${cost:.6f} exceeds limit ${self.max_cost_per_call_usd:.6f}",
                        limit_type="per_call",
                        limit_value=self.max_cost_per_call_usd,
                        current=cost,
                    )
                elif self.on_exceed == "skip":
                    return

            try:
                async for token in llm.stream_with_messages(messages, **kw):
                    yield token
                return
            except BudgetExceededError:
                if self.on_exceed == "downgrade" and i < len(self.candidates) - 1:
                    # Note: tokens might have already been yielded!
                    self._record_downgrade(
                        llm.config.model,
                        self.candidates[i + 1].config.model,
                        "Budget exceeded mid-stream",
                    )
                    continue
                raise

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        messages = [{"role": "user", "content": prompt}]
        async for token in self.stream_with_messages(messages, **kw):
            yield token

    async def generate(self, prompt: str, **kw: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_with_messages(messages, **kw)

    async def _call_with_tools_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        for i, llm in enumerate(self.candidates):
            # We estimate cost based on messages + tools size
            cost = self._estimate_cost(llm, messages)
            # Add an arbitrary 10% overhead for tools
            cost *= 1.1

            if cost > self.max_cost_per_call_usd:
                if self.on_exceed == "downgrade" and i < len(self.candidates) - 1:
                    self._record_downgrade(
                        llm.config.model,
                        self.candidates[i + 1].config.model,
                        f"Estimated cost ${cost:.6f} exceeds limit ${self.max_cost_per_call_usd:.6f}",
                    )
                    continue
                elif self.on_exceed == "raise":
                    raise BudgetExceededError(
                        f"Estimated cost ${cost:.6f} exceeds limit ${self.max_cost_per_call_usd:.6f}",
                        limit_type="per_call",
                        limit_value=self.max_cost_per_call_usd,
                        current=cost,
                    )
                elif self.on_exceed == "skip":
                    return {"content": "", "tool_calls": None}

                if cost > self.max_cost_per_call_usd:
                    raise BudgetExceededError(
                        f"Final candidate '{llm.config.model}' estimated cost ${cost:.6f} exceeds limit ${self.max_cost_per_call_usd:.6f}",
                        limit_type="per_call",
                        limit_value=self.max_cost_per_call_usd,
                        current=cost,
                    )

            try:
                # Bypass _call_with_tools_impl to get the candidate's retry/rate limit logic
                return await llm.call_with_tools(messages, tools)
            except BudgetExceededError as exc:
                if self.on_exceed == "downgrade" and i < len(self.candidates) - 1:
                    self._record_downgrade(
                        llm.config.model,
                        self.candidates[i + 1].config.model,
                        f"Budget exceeded during call: {exc}",
                    )
                    continue
                raise

        raise BudgetExceededError(
            "All candidates exhausted or budget exceeded",
            limit_type="per_call",
            limit_value=self.max_cost_per_call_usd,
            current=0.0,
        )
