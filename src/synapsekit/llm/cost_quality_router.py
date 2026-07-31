"""CostQualityRouter — learning-based routing: explore then exploit on observed cost/quality."""

from __future__ import annotations

import importlib
from collections.abc import AsyncGenerator
from typing import Any

from ..observability.tracer import COST_TABLE
from .base import BaseLLM, LLMConfig


class CostQualityRouter(BaseLLM):
    """Route requests to the cheapest model meeting a learned quality threshold.

    Phase 1 — Exploration: round-robin across all candidates for ``explore_n`` calls,
    collecting real cost and quality measurements.

    Phase 2 — Exploitation: route to the cheapest model whose observed ``avg_quality``
    meets ``quality_threshold``.  Falls back to the highest-quality model when none
    qualify, and respects ``budget_per_call_usd`` when provided.

    Example::

        router = CostQualityRouter(
            candidates=[cheap_llm, expensive_llm],
            quality_threshold=0.85,
            budget_per_call_usd=0.01,
            explore_n=50,
        )
        answer = await router.call("Summarise this document")
        print(router.stats())

    Grounding audit (#822)
    ----------------------
    ``GroundedSignal`` is **not applicable** here: the router's learning inputs
    are externally measured, not self-reported by the model being scored. Cost
    comes from token usage, and quality comes from a *separate* ``EvalSuite``
    evaluator via :meth:`_evaluate_quality` (or ``None``, which skips the
    update) — the candidate model never supplies its own score. The one caveat
    is that a user could configure the eval suite to be the same model as a
    candidate (LLM-as-judge on itself); that's a user eval-config decision, and
    even then it's a separate call/rubric, not the model grading its own routing
    observation. No provenance tag is threaded through as a result.
    """

    def __init__(
        self,
        candidates: list[BaseLLM],
        eval_suite: str | None = None,
        quality_threshold: float = 0.8,
        budget_per_call_usd: float | None = None,
        explore_n: int = 50,
    ) -> None:
        super().__init__(LLMConfig(model="__cq_router__", api_key="", provider="openai"))
        self._candidates = candidates
        self._eval_suite = eval_suite
        self._quality_threshold = quality_threshold
        self._budget_per_call_usd = budget_per_call_usd
        self._explore_n = explore_n

        self._calls = 0
        self._mode = "explore"

        # Keyed by model name; private _total_quality/_quality_calls drive the running avg.
        self._stats: dict[str, dict[str, Any]] = {
            llm.config.model: {
                "calls": 0,
                "avg_cost": 0.0,
                "avg_quality": 0.0,
                "_total_quality": 0.0,
                "_quality_calls": 0,
            }
            for llm in candidates
        }

        self._explore_index = 0
        self._evaluator: Any = None
        self._evaluator_loaded = False

    # ------------------------------------------------------------------ #
    # Eval suite integration
    # ------------------------------------------------------------------ #

    def _load_evaluator(self) -> Any:
        """Lazily import the eval_suite object (supports ``"pkg.mod:attr"`` or ``"pkg.mod.attr"``)."""
        if self._evaluator_loaded:
            return self._evaluator
        self._evaluator_loaded = True
        if not self._eval_suite:
            return None
        mod_path: str
        attr: str | None
        if ":" in self._eval_suite:
            mod_path, attr = self._eval_suite.rsplit(":", 1)
        else:
            parts = self._eval_suite.rsplit(".", 1)
            mod_path = parts[0]
            attr = parts[1] if len(parts) == 2 else None
        try:
            mod = importlib.import_module(mod_path)
            self._evaluator = getattr(mod, attr) if attr else mod
        except Exception:
            self._evaluator = None
        return self._evaluator

    async def _evaluate_quality(self, prompt: str, response: str) -> float | None:
        """Call the eval suite and return a score in [0, 1], or None if unavailable."""
        evaluator = self._load_evaluator()
        if evaluator is None:
            return None
        try:
            result = await evaluator.evaluate(question=prompt, answer=response)
            if hasattr(result, "mean_score"):
                score = result.mean_score
            elif isinstance(result, dict):
                score = result.get("score")
            elif isinstance(result, (int, float)):
                score = float(result)
            else:
                return None
            return float(score) if score is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Model selection
    # ------------------------------------------------------------------ #

    def _select_explore(self) -> BaseLLM:
        """Round-robin selection during exploration."""
        llm = self._candidates[self._explore_index % len(self._candidates)]
        self._explore_index += 1
        return llm

    def _exploit_ordering(self) -> list[BaseLLM]:
        """Return candidates ordered for exploitation.

        Priority:
          1. Unseen (calls == 0) OR meets quality threshold AND within budget — cheapest first
          2. Meets quality threshold but over budget    — cheapest first (quality-over-budget fallback)
          3. Below threshold                            — highest quality first (last resort)

        Unseen models land in group_a so early exploitation gives them a chance
        to accumulate stats rather than burying them behind proven-bad models.
        """
        group_a: list[BaseLLM] = []
        group_b: list[BaseLLM] = []
        group_c: list[BaseLLM] = []

        for llm in self._candidates:
            s = self._stats[llm.config.model]
            if s["calls"] == 0:
                group_a.append(llm)
                continue
            meets_quality = s["avg_quality"] >= self._quality_threshold
            within_budget = (
                self._budget_per_call_usd is None or s["avg_cost"] <= self._budget_per_call_usd
            )
            if meets_quality and within_budget:
                group_a.append(llm)
            elif meets_quality:
                group_b.append(llm)
            else:
                group_c.append(llm)

        def key_cost(m: BaseLLM) -> float:
            return float(self._stats[m.config.model]["avg_cost"])

        def key_quality(m: BaseLLM) -> float:
            return float(self._stats[m.config.model]["avg_quality"])

        group_a.sort(key=key_cost)
        group_b.sort(key=key_cost)
        group_c.sort(key=key_quality, reverse=True)

        return group_a + group_b + group_c

    # ------------------------------------------------------------------ #
    # Cost measurement
    # ------------------------------------------------------------------ #

    def _measure_cost(self, llm: BaseLLM, prev_in: int, prev_out: int) -> float:
        """Compute USD cost for the tokens consumed since the snapshot."""
        pricing = COST_TABLE.get(llm.config.model)
        if not pricing:
            return 0.0
        delta_in = max(0, llm._input_tokens - prev_in)
        delta_out = max(0, llm._output_tokens - prev_out)
        return delta_in * pricing["input"] + delta_out * pricing["output"]

    # ------------------------------------------------------------------ #
    # Stats update
    # ------------------------------------------------------------------ #

    def _update_stats(self, model: str, cost: float, quality: float | None) -> None:
        """Update running averages for cost and (optionally) quality."""
        s = self._stats[model]
        n = s["calls"]
        s["avg_cost"] = (s["avg_cost"] * n + cost) / (n + 1)
        if quality is not None:
            s["_total_quality"] += quality
            s["_quality_calls"] += 1
            s["avg_quality"] = s["_total_quality"] / s["_quality_calls"]
        s["calls"] = n + 1

    # ------------------------------------------------------------------ #
    # Core public API
    # ------------------------------------------------------------------ #

    async def call(self, prompt: str) -> str:
        """Route ``prompt`` to the best available model and return the response."""
        return await self.generate(prompt)

    async def generate(self, prompt: str, **kw: Any) -> str:  # type: ignore[override]
        """Select model, call, measure cost/quality, update stats, return response."""
        self._calls += 1
        is_explore = self._calls <= self._explore_n
        self._mode = "explore" if is_explore else "exploit"

        if is_explore:
            primary = self._select_explore()
            ordered = [primary] + [c for c in self._candidates if c is not primary]
        else:
            ordered = self._exploit_ordering()

        last_exc: Exception | None = None
        for llm in ordered:
            try:
                prev_in = llm._input_tokens
                prev_out = llm._output_tokens
                response = await llm.generate(prompt, **kw)
                if response is None:
                    response = ""
                cost = self._measure_cost(llm, prev_in, prev_out)
                quality = await self._evaluate_quality(prompt, response)
                self._update_stats(llm.config.model, cost, quality)
                return response
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No candidate models available")

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:  # type: ignore[override]
        """Stream from the selected model, updating stats after completion."""
        self._calls += 1
        is_explore = self._calls <= self._explore_n
        self._mode = "explore" if is_explore else "exploit"

        if is_explore:
            llm = self._select_explore()
        else:
            ordering = self._exploit_ordering()
            llm = ordering[0] if ordering else self._candidates[0]

        prev_in = llm._input_tokens
        prev_out = llm._output_tokens
        tokens: list[str] = []
        try:
            async for token in llm.stream(prompt, **kw):
                tokens.append(token)
                yield token
        except Exception:
            return

        response = "".join(tokens)
        cost = self._measure_cost(llm, prev_in, prev_out)
        quality = await self._evaluate_quality(prompt, response)
        self._update_stats(llm.config.model, cost, quality)

    # ------------------------------------------------------------------ #
    # Stats / Pareto frontier
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Return per-model stats and the Pareto frontier.

        Returns::

            {
                "models": {
                    model_name: {"avg_cost": float, "avg_quality": float, "calls": int}
                },
                "frontier": [{"model": str, "cost": float, "quality": float}, ...]
            }

        A model is on the Pareto frontier if no other model is both cheaper
        and higher quality.
        """
        models_info = {
            model: {
                "avg_cost": s["avg_cost"],
                "avg_quality": s["avg_quality"],
                "calls": s["calls"],
            }
            for model, s in self._stats.items()
        }
        return {"models": models_info, "frontier": self._pareto_frontier()}

    def _pareto_frontier(self) -> list[dict[str, Any]]:
        active = [(m, s) for m, s in self._stats.items() if s["calls"] > 0]
        frontier = []
        for model, s in active:
            dominated = any(
                os["avg_cost"] < s["avg_cost"] and os["avg_quality"] > s["avg_quality"]
                for om, os in active
                if om != model
            )
            if not dominated:
                frontier.append(
                    {
                        "model": model,
                        "cost": s["avg_cost"],
                        "quality": s["avg_quality"],
                    }
                )
        return frontier
