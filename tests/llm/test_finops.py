from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.llm.cost_quality_router import CostQualityRouter
from synapsekit.llm.fallback_chain import FallbackChain, FallbackChainConfig
from synapsekit.llm.finops import (
    BudgetLedger,
    BudgetPolicy,
    CarbonEstimator,
    CostArbitrageSimulator,
    ModelPricing,
    PricingTable,
    RequestClassPolicy,
    SimulationModel,
    SimulationTask,
)
from synapsekit.observability.budget_guard import BudgetExceededError
from synapsekit.observability.metrics import PrometheusMetrics


class MockLLM(BaseLLM):
    def __init__(self, model: str, provider: str = "openai", response: str = "ok") -> None:
        super().__init__(LLMConfig(model=model, api_key="test", provider=provider, max_retries=0))
        self.response = response
        self.calls = 0

    async def generate(self, prompt: str, **kw: Any) -> str:
        self.calls += 1
        self._input_tokens += 100
        self._output_tokens += 50
        return self.response

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
        self.calls += 1
        self._input_tokens += 100
        self._output_tokens += 50
        yield self.response


class FailingLLM(MockLLM):
    async def generate(self, prompt: str, **kw: Any) -> str:
        self.calls += 1
        raise TimeoutError("provider outage")

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str, None]:
        self.calls += 1
        raise TimeoutError("provider outage")
        yield ""


@pytest.mark.parametrize("cached", [0, 40])
def test_pricing_table_override_and_cached_input(cached: int) -> None:
    table = PricingTable(
        [
            ModelPricing(
                provider="openai",
                model="test-model",
                input_cost_per_million=2.0,
                output_cost_per_million=4.0,
                cached_input_cost_per_million=0.5,
                latency_sla_ms=120.0,
            )
        ]
    )
    pricing = table.get("test-model", provider="openai")
    assert pricing is not None
    expected_input = (100 - cached) * 2.0 / 1_000_000 + cached * 0.5 / 1_000_000
    assert pricing.estimate_cost(100, 50, cached_input_tokens=cached) == pytest.approx(
        expected_input + 50 * 4.0 / 1_000_000
    )


def test_pricing_table_provider_override_and_refresh() -> None:
    table = PricingTable.from_cost_table()
    replacement = table.override("openai", "gpt-4o", input_cost_per_million=1.0)
    assert replacement.provider == "openai"
    assert table.get("gpt-4o", provider="openai").input_cost_per_million == 1.0  # type: ignore[union-attr]

    refreshed = table.refresh(
        lambda: [
            ModelPricing(
                provider="openai",
                model="gpt-4o",
                input_cost_per_million=0.5,
                output_cost_per_million=2.0,
            )
        ]
    )
    assert refreshed == 1
    assert table.get("gpt-4o", provider="openai").input_cost_per_million == 0.5  # type: ignore[union-attr]


def test_budget_ledger_hard_caps_soft_alerts_and_attribution() -> None:
    alerts = []
    ledger = BudgetLedger(
        tenant_budgets={"tenant-a": BudgetPolicy(limit_usd=1.0, soft_alert_threshold=0.8)},
        key_budgets={"key-a": 1.0},
        on_alert=alerts.append,
    )

    reservation = ledger.reserve(0.8, tenant_id="tenant-a", api_key_id="key-a")
    ledger.commit(
        reservation,
        0.8,
        tenant_id="tenant-a",
        api_key_id="key-a",
        model="cheap",
        provider="test",
        input_tokens=10,
        output_tokens=5,
    )

    assert ledger.remaining_usd(tenant_id="tenant-a", api_key_id="key-a") == pytest.approx(0.2)
    assert ledger.attributions[0].tenant_id == "tenant-a"
    assert alerts and alerts[0].scope == "tenant"
    with pytest.raises(BudgetExceededError):
        ledger.check_before(0.3, tenant_id="tenant-a", api_key_id="key-a")


def test_budget_ledger_record_spend_enforces_cap_for_non_string_identifiers() -> None:
    ledger = BudgetLedger(tenant_budgets={"42": BudgetPolicy(limit_usd=1.0)})

    ledger.record_spend(
        0.9,
        tenant_id=42,
        model="cheap",
        provider="test",
        input_tokens=10,
        output_tokens=5,
    )

    assert ledger.remaining_usd(tenant_id=42) == pytest.approx(0.1)
    assert ledger.remaining_usd(tenant_id="42") == pytest.approx(0.1)
    with pytest.raises(BudgetExceededError):
        ledger.record_spend(
            0.5,
            tenant_id=42,
            model="cheap",
            provider="test",
            input_tokens=10,
            output_tokens=5,
        )


def _pricing() -> PricingTable:
    return PricingTable(
        [
            ModelPricing(
                provider="cheap-provider",
                model="cheap-model",
                input_cost_per_million=0.1,
                output_cost_per_million=0.1,
                latency_sla_ms=80.0,
                quality_score=0.92,
            ),
            ModelPricing(
                provider="best-provider",
                model="best-model",
                input_cost_per_million=4.0,
                output_cost_per_million=4.0,
                latency_sla_ms=200.0,
                quality_score=0.98,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_router_uses_cheapest_candidate_meeting_request_class_floor() -> None:
    cheap = MockLLM("cheap-model", provider="cheap-provider")
    best = MockLLM("best-model", provider="best-provider")
    router = CostQualityRouter(
        candidates=[best, cheap],
        explore_n=0,
        pricing_table=_pricing(),
        request_classes={"interactive": RequestClassPolicy(quality_floor=0.9, max_latency_ms=100)},
    )

    result = await router.generate("hello", request_class="interactive")

    assert result == "ok"
    assert cheap.calls == 1
    assert best.calls == 0
    assert router.last_route == "cheap-provider/cheap-model"


@pytest.mark.asyncio
async def test_router_budget_downgrades_and_records_savings() -> None:
    expensive = MockLLM("best-model", provider="best-provider")
    cheap = MockLLM("cheap-model", provider="cheap-provider")
    ledger = BudgetLedger(tenant_budgets={"tenant-a": 0.001})
    router = CostQualityRouter(
        candidates=[expensive, cheap],
        explore_n=0,
        pricing_table=_pricing(),
        budget_ledger=ledger,
    )

    result = await router.generate("hello", tenant_id="tenant-a")

    assert result == "ok"
    assert cheap.calls == 1
    assert expensive.calls == 0
    assert router.arbitrage_savings_usd > 0
    assert ledger.attributions[0].tenant_id == "tenant-a"


@pytest.mark.asyncio
async def test_router_composes_with_fallback_chain_for_outage() -> None:
    primary = FailingLLM("primary", provider="primary-provider")
    backup = MockLLM("backup", provider="backup-provider", response="backup response")
    chain = FallbackChain(FallbackChainConfig(models=[primary, backup]))
    router = CostQualityRouter(candidates=[chain], explore_n=0, pricing_table=PricingTable())

    assert await router.generate("hello") == "backup response"
    assert chain.used_model is backup


@pytest.mark.asyncio
async def test_router_stream_raises_like_generate_when_all_candidates_fail() -> None:
    only = FailingLLM("only", provider="only-provider")
    router = CostQualityRouter(candidates=[only], explore_n=0, pricing_table=PricingTable())

    with pytest.raises(TimeoutError, match="provider outage"):
        async for _ in router.stream("hello"):
            pass

    with pytest.raises(TimeoutError, match="provider outage"):
        await router.generate("hello")


def test_carbon_estimator_provider_and_region() -> None:
    estimator = CarbonEstimator({("cheap-provider", "eu-west"): 1.5})
    assert estimator.estimate("cheap-provider", "eu-west", 2_000) == pytest.approx(3.0)


def test_simulation_baseline_obeys_the_same_latency_sla() -> None:
    fast = SimulationModel(
        provider="fast-provider",
        model="fast-model",
        pricing=ModelPricing(
            provider="fast-provider",
            model="fast-model",
            input_cost_per_million=1.0,
            output_cost_per_million=1.0,
            latency_sla_ms=50,
        ),
        quality_by_class={"interactive": 0.9},
    )
    slow = SimulationModel(
        provider="slow-provider",
        model="slow-model",
        pricing=ModelPricing(
            provider="slow-provider",
            model="slow-model",
            input_cost_per_million=10.0,
            output_cost_per_million=10.0,
            latency_sla_ms=1_000,
        ),
        quality_by_class={"interactive": 0.99},
    )

    result = CostArbitrageSimulator([fast, slow]).replay(
        [SimulationTask("task", "interactive", 100, 100)],
        quality_floor=0.9,
        max_latency_ms=100,
    )

    assert result.routed_quality == result.baseline_quality == 0.9
    assert result.routed_cost_usd == result.baseline_cost_usd


def test_finops_prometheus_methods_are_disabled_safe() -> None:
    metrics = PrometheusMetrics(enabled=False)
    metrics.record_budget_remaining(tenant_id="tenant-a", api_key_id="key-a", remaining_usd=1.0)
    metrics.record_arbitrage_savings(
        provider="cheap-provider", model="cheap-model", savings_usd=0.5
    )
    assert metrics.enabled is False


def test_budget_ledger_never_commits_actual_cost_over_hard_cap() -> None:
    ledger = BudgetLedger(tenant_budgets={"tenant-a": 1.0})
    reservation = ledger.reserve(0.5, tenant_id="tenant-a")

    with pytest.raises(BudgetExceededError):
        ledger.commit(
            reservation,
            1.5,
            model="model",
            provider="provider",
            input_tokens=10,
            output_tokens=10,
        )

    assert ledger.spend_usd(tenant_id="tenant-a") == 0.0
    assert ledger.remaining_usd(tenant_id="tenant-a") == pytest.approx(0.5)
    ledger.release(reservation)
    assert ledger.remaining_usd(tenant_id="tenant-a") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="inactive"):
        ledger.commit(
            reservation,
            0.1,
            model="model",
            provider="provider",
            input_tokens=10,
            output_tokens=10,
        )


@pytest.mark.asyncio
async def test_router_uses_current_provider_price_for_duplicate_model_names() -> None:
    costly = MockLLM("shared-model", provider="costly-provider")
    cheap = MockLLM("shared-model", provider="cheap-provider")
    pricing = PricingTable(
        [
            ModelPricing(
                provider="costly-provider",
                model="shared-model",
                input_cost_per_million=10.0,
                output_cost_per_million=10.0,
                quality_score=0.95,
            ),
            ModelPricing(
                provider="cheap-provider",
                model="shared-model",
                input_cost_per_million=1.0,
                output_cost_per_million=1.0,
                quality_score=0.95,
            ),
        ]
    )
    router = CostQualityRouter(candidates=[costly, cheap], explore_n=1, pricing_table=pricing)

    await router.generate("first")
    await router.generate("second")

    assert costly.calls == 1
    assert cheap.calls == 1
    assert router.last_route == "cheap-provider/shared-model"


@pytest.mark.asyncio
async def test_router_rejects_candidate_without_required_latency_sla() -> None:
    candidate = MockLLM("unknown-latency", provider="provider")
    router = CostQualityRouter(
        candidates=[candidate],
        explore_n=0,
        pricing_table=PricingTable(
            [
                ModelPricing(
                    provider="provider",
                    model="unknown-latency",
                    input_cost_per_million=1.0,
                    output_cost_per_million=1.0,
                    quality_score=0.95,
                )
            ]
        ),
        request_classes={"interactive": RequestClassPolicy(quality_floor=0.9, max_latency_ms=100)},
    )

    with pytest.raises(RuntimeError, match="quality/latency"):
        await router.generate("hello", request_class="interactive")

    assert candidate.calls == 0


@pytest.mark.asyncio
async def test_router_applies_request_policy_inside_fallback_chain() -> None:
    slow = MockLLM("slow", provider="provider", response="slow")
    fast = MockLLM("fast", provider="provider", response="fast")
    chain = FallbackChain(FallbackChainConfig(models=[slow, fast]))
    router = CostQualityRouter(
        candidates=[chain],
        explore_n=0,
        pricing_table=PricingTable(
            [
                ModelPricing(
                    provider="provider",
                    model="slow",
                    input_cost_per_million=1.0,
                    output_cost_per_million=1.0,
                    latency_sla_ms=1_000,
                    quality_score=0.5,
                ),
                ModelPricing(
                    provider="provider",
                    model="fast",
                    input_cost_per_million=1.0,
                    output_cost_per_million=1.0,
                    latency_sla_ms=50,
                    quality_score=0.95,
                ),
            ]
        ),
        request_classes={"interactive": RequestClassPolicy(quality_floor=0.9, max_latency_ms=100)},
    )

    assert await router.generate("hello", request_class="interactive") == "fast"
    assert slow.calls == 0
    assert fast.calls == 1


@pytest.mark.asyncio
async def test_router_stream_applies_request_policy_inside_fallback_chain() -> None:
    slow = MockLLM("slow", provider="provider", response="slow")
    fast = MockLLM("fast", provider="provider", response="fast")
    chain = FallbackChain(FallbackChainConfig(models=[slow, fast]))
    router = CostQualityRouter(
        candidates=[chain],
        explore_n=0,
        pricing_table=PricingTable(
            [
                ModelPricing(
                    provider="provider",
                    model="slow",
                    input_cost_per_million=1.0,
                    output_cost_per_million=1.0,
                    latency_sla_ms=1_000,
                    quality_score=0.5,
                ),
                ModelPricing(
                    provider="provider",
                    model="fast",
                    input_cost_per_million=1.0,
                    output_cost_per_million=1.0,
                    latency_sla_ms=50,
                    quality_score=0.95,
                ),
            ]
        ),
        request_classes={"interactive": RequestClassPolicy(quality_floor=0.9, max_latency_ms=100)},
    )

    tokens = [token async for token in router.stream("hello", request_class="interactive")]

    assert tokens == ["fast"]
    assert slow.calls == 0
    assert fast.calls == 1


@pytest.mark.asyncio
async def test_router_rejects_unpriced_model_when_tenant_budget_is_enforced() -> None:
    candidate = MockLLM("unpriced", provider="provider")
    router = CostQualityRouter(
        candidates=[candidate],
        explore_n=0,
        pricing_table=PricingTable(),
        budget_ledger=BudgetLedger(tenant_budgets={"tenant-a": 1.0}),
    )

    with pytest.raises(BudgetExceededError, match="pricing"):
        await router.generate("hello", tenant_id="tenant-a")

    assert candidate.calls == 0


@pytest.mark.asyncio
async def test_router_generate_still_returns_response_when_actual_spend_settles_over_a_hard_cap() -> (
    None
):
    candidate = MockLLM("priced", provider="provider")
    ledger = BudgetLedger(tenant_budgets={"tenant-a": 0.001})
    router = CostQualityRouter(
        candidates=[candidate],
        explore_n=0,
        budget_ledger=ledger,
        pricing_table=PricingTable(
            [
                ModelPricing(
                    provider="provider",
                    model="priced",
                    input_cost_per_million=10.0,
                    output_cost_per_million=10.0,
                    quality_score=0.95,
                )
            ]
        ),
    )

    # The provider call already succeeded by the time settlement discovers
    # the actual cost overshoots the cap — the response must not be
    # discarded, but the overage is still recorded against the ledger.
    result = await router.generate("hello", tenant_id="tenant-a", max_tokens=1)

    assert result == "ok"
    assert candidate.calls == 1
    assert ledger.spend_usd(tenant_id="tenant-a") > 0.001
    assert ledger.remaining_usd(tenant_id="tenant-a") == 0.0


@pytest.mark.asyncio
async def test_router_stream_still_yields_tokens_when_actual_spend_settles_over_a_hard_cap() -> (
    None
):
    candidate = MockLLM("priced", provider="provider")
    ledger = BudgetLedger(tenant_budgets={"tenant-a": 0.001})
    router = CostQualityRouter(
        candidates=[candidate],
        explore_n=0,
        budget_ledger=ledger,
        pricing_table=PricingTable(
            [
                ModelPricing(
                    provider="provider",
                    model="priced",
                    input_cost_per_million=10.0,
                    output_cost_per_million=10.0,
                    quality_score=0.95,
                )
            ]
        ),
    )

    tokens = [token async for token in router.stream("hello", tenant_id="tenant-a", max_tokens=1)]

    assert tokens == ["ok"]
    assert candidate.calls == 1
    assert ledger.spend_usd(tenant_id="tenant-a") > 0.001
    assert ledger.remaining_usd(tenant_id="tenant-a") == 0.0
