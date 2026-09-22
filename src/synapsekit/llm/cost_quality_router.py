"""CostQualityRouter — learning-based routing: explore then exploit on observed cost/quality."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from ..observability.budget_guard import BudgetExceededError
from .base import BaseLLM, LLMConfig, _messages_to_prompt
from .cost_router import QUALITY_TABLE
from .finops import (
    BudgetLedger,
    CarbonEstimator,
    ModelPricing,
    PricingTable,
    RequestClassPolicy,
)

logger = logging.getLogger(__name__)


class CostQualityRouter(BaseLLM):
    """Learning router with provider price/SLA arbitrage and budget governance.

    The original explore/exploit API is retained. Optional FinOps metadata is
    passed to ``generate``/``stream`` as keyword arguments and removed before
    the selected provider is called: ``request_class``, ``quality_floor``,
    ``max_latency_ms``, ``tenant_id``, ``api_key_id`` (or ``key_id``),
    ``region``, and ``cached_input_tokens``.
    """

    def __init__(
        self,
        candidates: list[BaseLLM],
        eval_suite: str | None = None,
        quality_threshold: float = 0.8,
        budget_per_call_usd: float | None = None,
        explore_n: int = 50,
        *,
        pricing_table: PricingTable | None = None,
        request_classes: dict[str, RequestClassPolicy] | None = None,
        budget_ledger: BudgetLedger | None = None,
        metrics: Any | None = None,
        carbon_estimator: CarbonEstimator | None = None,
    ) -> None:
        super().__init__(LLMConfig(model="__cq_router__", api_key="", provider="router"))
        self._candidates = list(candidates)
        self._eval_suite = eval_suite
        self._quality_threshold = quality_threshold
        self._budget_per_call_usd = budget_per_call_usd
        self._explore_n = max(0, explore_n)
        self._pricing_table = pricing_table or PricingTable.from_cost_table(
            quality_scores=QUALITY_TABLE
        )
        self._request_classes = dict(request_classes or {})
        self._budget_ledger = budget_ledger
        self._metrics = metrics
        self._carbon_estimator = carbon_estimator
        self._calls = 0
        self._mode = "explore"
        self._explore_index = 0
        self._selected_model: str | None = None
        self._last_route: str | None = None
        self._last_cost_usd = 0.0
        self._last_carbon_grams: float | None = None
        self._arbitrage_savings_usd = 0.0
        self._pending_events: list[dict[str, Any]] = []
        self._evaluator: Any = None
        self._evaluator_loaded = False
        self._stats: dict[str, dict[str, Any]] = {
            self._stats_key(llm): {
                "calls": 0,
                "avg_cost": 0.0,
                "avg_quality": 0.0,
                "_total_quality": 0.0,
                "_quality_calls": 0,
            }
            for llm in self._candidates
        }

    def _stats_key(self, llm: BaseLLM) -> str:
        """Keep legacy model-only stats unless model names collide across providers."""
        matches = sum(candidate.config.model == llm.config.model for candidate in self._candidates)
        return f"{llm.config.provider}/{llm.config.model}" if matches > 1 else llm.config.model

    def _stats_for(self, llm: BaseLLM) -> dict[str, Any]:
        key = self._stats_key(llm)
        if key not in self._stats:
            self._stats[key] = {
                "calls": 0,
                "avg_cost": 0.0,
                "avg_quality": 0.0,
                "_total_quality": 0.0,
                "_quality_calls": 0,
            }
        return self._stats[key]

    @staticmethod
    def _fallback_models(candidate: BaseLLM) -> list[BaseLLM] | None:
        config = getattr(candidate, "_chain_config", None)
        models = getattr(config, "models", None)
        return list(models) if isinstance(models, list) else None

    def _eligible_fallback_models(
        self,
        candidate: BaseLLM,
        policy: RequestClassPolicy,
        region: str | None,
    ) -> list[BaseLLM] | None:
        models = self._fallback_models(candidate)
        if models is None:
            return None
        return [model for model in models if self._matches(model, policy, region)]

    def _expected_llm(
        self,
        candidate: BaseLLM,
        policy: RequestClassPolicy,
        region: str | None,
    ) -> BaseLLM:
        eligible = self._eligible_fallback_models(candidate, policy, region)
        return eligible[0] if eligible else candidate

    def _provider_kwargs(
        self,
        candidate: BaseLLM,
        provider_kw: dict[str, Any],
        policy: RequestClassPolicy,
        region: str | None,
        *,
        enforce_policy: bool,
    ) -> dict[str, Any]:
        if not enforce_policy:
            return provider_kw
        eligible = self._eligible_fallback_models(candidate, policy, region)
        if eligible is None:
            return provider_kw
        forwarded = dict(provider_kw)
        forwarded["_synapsekit_allowed_models"] = eligible
        return forwarded

    def _load_evaluator(self) -> Any:
        if self._evaluator_loaded:
            return self._evaluator
        self._evaluator_loaded = True
        if not self._eval_suite:
            return None
        module_path: str
        attribute: str | None
        if ":" in self._eval_suite:
            module_path, attribute = self._eval_suite.rsplit(":", 1)
        else:
            parts = self._eval_suite.rsplit(".", 1)
            module_path, attribute = parts[0], parts[1] if len(parts) == 2 else None
        try:
            module = importlib.import_module(module_path)
            self._evaluator = getattr(module, attribute) if attribute else module
        except Exception:
            self._evaluator = None
        return self._evaluator

    async def _evaluate_quality(self, prompt: str, response: str) -> float | None:
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

    def _resolve_policy(
        self,
        request_class: str | RequestClassPolicy | None,
        quality_floor: float | None,
        max_latency_ms: float | None,
    ) -> tuple[RequestClassPolicy, bool, str | None]:
        explicit = (
            request_class is not None or quality_floor is not None or max_latency_ms is not None
        )
        if isinstance(request_class, RequestClassPolicy):
            policy = request_class
            name = None
        elif isinstance(request_class, str):
            if request_class not in self._request_classes:
                raise KeyError(f"unknown request class: {request_class!r}")
            policy = self._request_classes[request_class]
            name = request_class
        else:
            policy = RequestClassPolicy(quality_floor=self._quality_threshold)
            name = None
        if quality_floor is not None or max_latency_ms is not None:
            policy = RequestClassPolicy(
                quality_floor=quality_floor if quality_floor is not None else policy.quality_floor,
                max_latency_ms=max_latency_ms
                if max_latency_ms is not None
                else policy.max_latency_ms,
            )
        return policy, explicit, name

    def _pricing(self, llm: BaseLLM, region: str | None = None) -> ModelPricing | None:
        return self._pricing_table.get(
            llm.config.model,
            provider=llm.config.provider,
            region=region,
        )

    def _quality_for(self, llm: BaseLLM, region: str | None = None) -> float:
        fallback_models = self._fallback_models(llm)
        if fallback_models is not None:
            return max((self._quality_for(model, region) for model in fallback_models), default=0.0)
        stats = self._stats.get(self._stats_key(llm))
        if stats and stats["_quality_calls"] > 0:
            return float(stats["avg_quality"])
        pricing = self._pricing(llm, region)
        if pricing is not None and pricing.quality_score is not None:
            return pricing.quality_score
        return QUALITY_TABLE.get(llm.config.model, 0.5)

    def _matches(self, llm: BaseLLM, policy: RequestClassPolicy, region: str | None) -> bool:
        fallback_models = self._fallback_models(llm)
        if fallback_models is not None:
            return any(self._matches(model, policy, region) for model in fallback_models)
        if self._quality_for(llm, region) < policy.quality_floor:
            return False
        pricing = self._pricing(llm, region)
        if policy.max_latency_ms is None:
            return True
        return (
            pricing is not None
            and pricing.latency_sla_ms is not None
            and (pricing.latency_sla_ms <= policy.max_latency_ms)
        )

    def _static_cost(self, llm: BaseLLM, region: str | None) -> float:
        pricing = self._pricing(llm, region)
        return pricing.estimate_cost(1_000, 1_000) if pricing is not None else float("inf")

    def _candidate_order(
        self,
        policy: RequestClassPolicy,
        *,
        strict: bool,
        region: str | None,
        input_tokens: int = 1_000,
        output_tokens: int = 1_024,
    ) -> list[BaseLLM]:
        pool = [
            candidate for candidate in self._candidates if self._matches(candidate, policy, region)
        ]
        if not pool and not strict:
            pool = list(self._candidates)
        if not pool:
            return []

        group_a: list[BaseLLM] = []
        group_b: list[BaseLLM] = []
        group_c: list[BaseLLM] = []
        for candidate in pool:
            stats = self._stats_for(candidate)
            if stats["calls"] == 0:
                group_a.append(candidate)
                continue
            meets_quality = self._quality_for(candidate, region) >= policy.quality_floor
            current_cost = self._estimate_cost(
                candidate,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=0,
                policy=policy,
                region=region,
            )
            within_budget = self._budget_per_call_usd is None or (
                current_cost is not None and current_cost <= self._budget_per_call_usd
            )
            if meets_quality and within_budget:
                group_a.append(candidate)
            elif meets_quality:
                group_b.append(candidate)
            else:
                group_c.append(candidate)

        def cost(candidate: BaseLLM) -> float:
            current_cost = self._estimate_cost(
                candidate,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=0,
                policy=policy,
                region=region,
            )
            if current_cost is not None:
                return current_cost
            stats = self._stats_for(candidate)
            return float(stats["avg_cost"]) if stats["calls"] > 0 else float("inf")

        group_a.sort(key=cost)
        group_b.sort(key=cost)
        group_c.sort(key=lambda candidate: self._quality_for(candidate, region), reverse=True)
        return group_a + group_b + group_c

    def _explore_candidate(
        self, policy: RequestClassPolicy, *, strict: bool, region: str | None
    ) -> BaseLLM | None:
        pool = [
            candidate for candidate in self._candidates if self._matches(candidate, policy, region)
        ]
        if not pool and not strict:
            pool = list(self._candidates)
        if not pool:
            return None
        candidate = pool[self._explore_index % len(pool)]
        self._explore_index += 1
        return candidate

    @staticmethod
    def _metadata(kw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        provider_kw = dict(kw)
        metadata: dict[str, Any] = {}
        for key in (
            "request_class",
            "quality_floor",
            "max_latency_ms",
            "tenant_id",
            "api_key_id",
            "region",
            "cached_input_tokens",
        ):
            if key in provider_kw:
                metadata[key] = provider_kw.pop(key)
        if metadata.get("api_key_id") is None and "key_id" in provider_kw:
            metadata["api_key_id"] = provider_kw.pop("key_id")
        metadata["cached_input_tokens"] = max(0, int(metadata.get("cached_input_tokens", 0) or 0))
        return provider_kw, metadata

    @staticmethod
    def _estimate_input_tokens(prompt: str) -> int:
        return max(1, (len(prompt) + 3) // 4) if prompt else 0

    def _estimate_cost(
        self,
        candidate: BaseLLM,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
        policy: RequestClassPolicy,
        region: str | None,
    ) -> float | None:
        pricing = self._pricing(self._expected_llm(candidate, policy, region), region)
        if pricing is None:
            return None
        return pricing.estimate_cost(
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    @staticmethod
    def _effective_llm(candidate: BaseLLM) -> BaseLLM:
        used_model = getattr(candidate, "used_model", None)
        return used_model if isinstance(used_model, BaseLLM) else candidate

    @staticmethod
    def _token_snapshot(llm: BaseLLM) -> tuple[int, int]:
        effective = CostQualityRouter._effective_llm(llm)
        return effective._input_tokens, effective._output_tokens

    def _measure_cost(
        self,
        candidate: BaseLLM,
        previous: tuple[int, int],
        *,
        cached_input_tokens: int,
        region: str | None,
    ) -> tuple[float, int, int, BaseLLM]:
        effective = self._effective_llm(candidate)
        input_tokens = max(0, effective._input_tokens - previous[0])
        output_tokens = max(0, effective._output_tokens - previous[1])
        pricing = self._pricing(effective, region) or self._pricing(candidate, region)
        cost = (
            pricing.estimate_cost(
                input_tokens,
                output_tokens,
                cached_input_tokens=cached_input_tokens,
            )
            if pricing is not None
            else 0.0
        )
        return cost, input_tokens, output_tokens, effective

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
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def _baseline_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        policy: RequestClassPolicy,
        region: str | None,
        cached_input_tokens: int,
    ) -> float:
        if not self._candidates:
            return 0.0
        compatible = [
            candidate for candidate in self._candidates if self._matches(candidate, policy, region)
        ]
        baseline = max(
            compatible or self._candidates,
            key=lambda candidate: self._quality_for(candidate, region),
        )
        expected = self._expected_llm(baseline, policy, region)
        pricing = self._pricing(self._effective_llm(expected), region)
        return (
            pricing.estimate_cost(
                input_tokens,
                output_tokens,
                cached_input_tokens=cached_input_tokens,
            )
            if pricing is not None
            else 0.0
        )

    async def _record_result(
        self,
        candidate: BaseLLM,
        prompt: str,
        response: str,
        previous: tuple[int, int],
        *,
        policy: RequestClassPolicy,
        request_class: str | None,
        tenant_id: str | None,
        api_key_id: str | None,
        region: str | None,
        cached_input_tokens: int,
        started_at: float,
        reservation: Any,
    ) -> str:
        cost, input_tokens, output_tokens, effective = self._measure_cost(
            candidate,
            previous,
            cached_input_tokens=cached_input_tokens,
            region=region,
        )
        quality = await self._evaluate_quality(prompt, response)
        self._update_stats(candidate, cost, quality)
        provider = effective.config.provider
        model = effective.config.model
        carbon = (
            self._carbon_estimator.estimate(provider, region, input_tokens + output_tokens)
            if self._carbon_estimator is not None
            else None
        )
        if self._budget_ledger is not None:
            commit_kwargs: dict[str, Any] = {
                "model": model,
                "provider": provider,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tenant_id": tenant_id,
                "api_key_id": api_key_id,
                "request_class": request_class,
                "carbon_grams": carbon,
            }
            try:
                self._budget_ledger.commit(reservation, cost, **commit_kwargs)
            except BudgetExceededError as exc:
                # The provider call already succeeded and cannot be undone —
                # record the real spend (even though it breaches the cap)
                # instead of discarding a response the caller already paid for.
                logger.warning(
                    "Budget settlement exceeded cap for %s/%s after a successful call; "
                    "recording actual spend of $%.6f anyway: %s",
                    provider,
                    model,
                    cost,
                    exc,
                )
                self._budget_ledger.commit(reservation, cost, force=True, **commit_kwargs)
        self._selected_model = model
        self._last_route = f"{provider}/{model}"
        self._last_cost_usd = cost
        self._last_carbon_grams = carbon
        baseline_cost = self._baseline_cost(
            input_tokens,
            output_tokens,
            policy=policy,
            region=region,
            cached_input_tokens=cached_input_tokens,
        )
        self._arbitrage_savings_usd += baseline_cost - cost
        if self._metrics is not None:
            self._metrics.record_llm(
                model=model,
                provider=provider,
                cost_usd=cost,
                total_tokens=input_tokens + output_tokens,
                latency_ms=(time.monotonic() - started_at) * 1_000,
            )
            self._metrics.record_arbitrage_savings(
                provider=provider,
                model=model,
                savings_usd=self._arbitrage_savings_usd,
            )
            if self._budget_ledger is not None:
                remaining = self._budget_ledger.remaining_usd(
                    tenant_id=tenant_id,
                    api_key_id=api_key_id,
                )
                if remaining is not None:
                    self._metrics.record_budget_remaining(
                        tenant_id=tenant_id,
                        api_key_id=api_key_id,
                        remaining_usd=remaining,
                    )
        return response

    def _update_stats(self, candidate: BaseLLM, cost: float, quality: float | None) -> None:
        stats = self._stats_for(candidate)
        calls = stats["calls"]
        stats["avg_cost"] = (stats["avg_cost"] * calls + cost) / (calls + 1)
        if quality is not None:
            stats["_total_quality"] += quality
            stats["_quality_calls"] += 1
            stats["avg_quality"] = stats["_total_quality"] / stats["_quality_calls"]
        stats["calls"] = calls + 1

    async def _run_generate(self, prompt: str, invoke: Any, kw: dict[str, Any]) -> str:
        provider_kw, metadata = self._metadata(kw)
        policy, strict, request_class = self._resolve_policy(
            metadata.get("request_class"),
            metadata.get("quality_floor"),
            metadata.get("max_latency_ms"),
        )
        tenant_id = metadata.get("tenant_id")
        api_key_id = metadata.get("api_key_id")
        region = metadata.get("region")
        cached_input_tokens = metadata["cached_input_tokens"]
        self._calls += 1
        is_explore = self._calls <= self._explore_n
        self._mode = "explore" if is_explore else "exploit"
        input_estimate = self._estimate_input_tokens(prompt)
        output_estimate = int(provider_kw.get("max_tokens", 0) or 1_024)
        if is_explore:
            primary = self._explore_candidate(policy, strict=strict, region=region)
            ordered = (
                [primary]
                + [candidate for candidate in self._candidates if candidate is not primary]
                if primary is not None
                else []
            )
            if strict:
                ordered = [
                    candidate for candidate in ordered if self._matches(candidate, policy, region)
                ]
        else:
            ordered = self._candidate_order(
                policy,
                strict=strict,
                region=region,
                input_tokens=input_estimate,
                output_tokens=output_estimate,
            )
        if not ordered:
            raise RuntimeError("No candidate satisfies the request quality/latency policy")

        within_budget: list[BaseLLM] = []
        over_budget: list[BaseLLM] = []
        for candidate in ordered:
            estimate = self._estimate_cost(
                candidate,
                input_tokens=input_estimate,
                output_tokens=output_estimate,
                cached_input_tokens=cached_input_tokens,
                policy=policy,
                region=region,
            )
            if (
                self._budget_per_call_usd is not None
                and estimate is not None
                and estimate > self._budget_per_call_usd
            ):
                over_budget.append(candidate)
            else:
                within_budget.append(candidate)
        ordered = within_budget + over_budget
        governed = self._budget_ledger is not None and self._budget_ledger.has_budget(
            tenant_id=tenant_id, api_key_id=api_key_id
        )
        last_exc: Exception | None = None
        for index, candidate in enumerate(ordered):
            estimate = self._estimate_cost(
                candidate,
                input_tokens=input_estimate,
                output_tokens=output_estimate,
                cached_input_tokens=cached_input_tokens,
                policy=policy,
                region=region,
            )
            reservation = None
            try:
                if estimate is None and governed:
                    raise BudgetExceededError(
                        "Verified provider pricing is required when a tenant or API-key budget is enforced",
                        "pricing",
                        0.0,
                        0.0,
                    )
                if self._budget_ledger is not None:
                    reservation = self._budget_ledger.reserve(
                        estimate or 0.0,
                        tenant_id=tenant_id,
                        api_key_id=api_key_id,
                    )
                previous = self._token_snapshot(candidate)
                started_at = time.monotonic()
                result = await invoke(
                    candidate,
                    self._provider_kwargs(
                        candidate, provider_kw, policy, region, enforce_policy=strict
                    ),
                )
                return await self._record_result(
                    candidate,
                    prompt,
                    str(result) if result is not None else "",
                    previous,
                    policy=policy,
                    request_class=request_class,
                    tenant_id=tenant_id,
                    api_key_id=api_key_id,
                    region=region,
                    cached_input_tokens=cached_input_tokens,
                    started_at=started_at,
                    reservation=reservation,
                )
            except Exception as exc:
                if reservation is not None and self._budget_ledger is not None:
                    self._budget_ledger.release(reservation)
                last_exc = exc
                if index + 1 < len(ordered):
                    self._record_downgrade(
                        candidate.config.model,
                        ordered[index + 1].config.model,
                        str(exc),
                    )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No candidate models available")

    async def call(self, prompt: str, **kw: Any) -> str:
        return await self.generate(prompt, **kw)

    async def generate(self, prompt: str, **kw: Any) -> str:
        return await self._run_generate(
            prompt,
            lambda candidate, provider_kw: candidate.generate(prompt, **provider_kw),
            kw,
        )

    async def generate_with_messages(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        prompt = _messages_to_prompt(messages)
        return await self._run_generate(
            prompt,
            lambda candidate, provider_kw: candidate.generate_with_messages(
                messages, **provider_kw
            ),
            kw,
        )

    async def _stream_generate(
        self,
        prompt: str,
        invoke: Any,
        kw: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        provider_kw, metadata = self._metadata(kw)
        policy, strict, request_class = self._resolve_policy(
            metadata.get("request_class"),
            metadata.get("quality_floor"),
            metadata.get("max_latency_ms"),
        )
        tenant_id = metadata.get("tenant_id")
        api_key_id = metadata.get("api_key_id")
        region = metadata.get("region")
        cached_input_tokens = metadata["cached_input_tokens"]
        self._calls += 1
        is_explore = self._calls <= self._explore_n
        self._mode = "explore" if is_explore else "exploit"
        if is_explore:
            primary = self._explore_candidate(policy, strict=strict, region=region)
            ordered = (
                [primary]
                + [candidate for candidate in self._candidates if candidate is not primary]
                if primary is not None
                else []
            )
            if strict:
                ordered = [
                    candidate for candidate in ordered if self._matches(candidate, policy, region)
                ]
        else:
            ordered = self._candidate_order(policy, strict=strict, region=region)
        if not ordered:
            raise RuntimeError("No candidate satisfies the request quality/latency policy")
        input_estimate = self._estimate_input_tokens(prompt)
        output_estimate = int(provider_kw.get("max_tokens", 0) or 1_024)
        governed = self._budget_ledger is not None and self._budget_ledger.has_budget(
            tenant_id=tenant_id, api_key_id=api_key_id
        )
        last_exc: Exception | None = None
        for index, candidate in enumerate(ordered):
            estimate = self._estimate_cost(
                candidate,
                input_tokens=input_estimate,
                output_tokens=output_estimate,
                cached_input_tokens=cached_input_tokens,
                policy=policy,
                region=region,
            )
            reservation = None
            tokens: list[str] = []
            try:
                if estimate is None and governed:
                    raise BudgetExceededError(
                        "Verified provider pricing is required when a tenant or API-key budget is enforced",
                        "pricing",
                        0.0,
                        0.0,
                    )
                if (
                    self._budget_per_call_usd is not None
                    and estimate is not None
                    and estimate > self._budget_per_call_usd
                ):
                    raise BudgetExceededError(
                        f"Estimated cost ${estimate:.6f} exceeds per-call limit ${self._budget_per_call_usd:.6f}",
                        "per_call",
                        self._budget_per_call_usd,
                        estimate,
                    )
                if self._budget_ledger is not None:
                    reservation = self._budget_ledger.reserve(
                        estimate or 0.0,
                        tenant_id=tenant_id,
                        api_key_id=api_key_id,
                    )
                previous = self._token_snapshot(candidate)
                started_at = time.monotonic()
                async for token in invoke(
                    candidate,
                    self._provider_kwargs(
                        candidate, provider_kw, policy, region, enforce_policy=strict
                    ),
                ):
                    tokens.append(str(token))
                    yield str(token)
                await self._record_result(
                    candidate,
                    prompt,
                    "".join(tokens),
                    previous,
                    policy=policy,
                    request_class=request_class,
                    tenant_id=tenant_id,
                    api_key_id=api_key_id,
                    region=region,
                    cached_input_tokens=cached_input_tokens,
                    started_at=started_at,
                    reservation=reservation,
                )
                return
            except Exception as exc:
                if reservation is not None and self._budget_ledger is not None:
                    self._budget_ledger.release(reservation)
                if tokens:
                    raise
                last_exc = exc
                if index + 1 < len(ordered):
                    self._record_downgrade(
                        candidate.config.model,
                        ordered[index + 1].config.model,
                        str(exc),
                    )
        if last_exc is not None:
            raise last_exc
        return

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
        async for token in self._stream_generate(
            prompt,
            lambda candidate, provider_kw: candidate.stream(prompt, **provider_kw),
            kw,
        ):
            yield token

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str, None]:
        prompt = _messages_to_prompt(messages)
        async for token in self._stream_generate(
            prompt,
            lambda candidate, provider_kw: candidate.stream_with_messages(messages, **provider_kw),
            kw,
        ):
            yield token

    async def _call_with_tools_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = _messages_to_prompt(messages)
        result = await self._run_generate(
            prompt,
            lambda candidate, _provider_kw: candidate.call_with_tools(messages, tools),
            {},
        )
        return {"content": result, "tool_calls": None}

    @property
    def selected_model(self) -> str | None:
        return self._selected_model

    @property
    def last_route(self) -> str | None:
        return self._last_route

    @property
    def last_cost_usd(self) -> float:
        return self._last_cost_usd

    @property
    def last_carbon_grams(self) -> float | None:
        return self._last_carbon_grams

    @property
    def arbitrage_savings_usd(self) -> float:
        return self._arbitrage_savings_usd

    def stats(self) -> dict[str, Any]:
        models_info = {
            model: {
                "avg_cost": stats["avg_cost"],
                "avg_quality": stats["avg_quality"],
                "calls": stats["calls"],
            }
            for model, stats in self._stats.items()
        }
        active = [(model, stats) for model, stats in self._stats.items() if stats["calls"] > 0]
        frontier: list[dict[str, Any]] = []
        for model, stats in active:
            dominated = any(
                other["avg_cost"] < stats["avg_cost"]
                and other["avg_quality"] > stats["avg_quality"]
                for other_model, other in active
                if other_model != model
            )
            if not dominated:
                frontier.append(
                    {
                        "model": model,
                        "cost": stats["avg_cost"],
                        "quality": stats["avg_quality"],
                    }
                )
        return {"models": models_info, "frontier": frontier}
