from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from synapsekit.agents.executor import AgentConfig, AgentExecutor
from synapsekit.agents.step_events import CostDowngradeEvent
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.llm.cost_router import CostQualityRouter
from synapsekit.observability.budget_guard import BudgetExceededError


class _MockLLM(BaseLLM):
    def __init__(self, model: str, cost_to_raise: float | None = None):
        super().__init__(LLMConfig(model=model, api_key="test", provider="openai"))
        self.cost_to_raise = cost_to_raise

    async def generate_with_messages(self, messages: list[dict], **kw) -> str:
        if self.cost_to_raise is not None:
            raise BudgetExceededError("Too expensive", "per_call", 0.01, self.cost_to_raise)
        return f"Response from {self.config.model}"

    async def stream_with_messages(self, messages: list[dict], **kw) -> AsyncGenerator[str]:
        if self.cost_to_raise is not None:
            raise BudgetExceededError("Too expensive", "per_call", 0.01, self.cost_to_raise)
        yield f"Response from {self.config.model}"

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        if self.cost_to_raise is not None:
            raise BudgetExceededError("Too expensive", "per_call", 0.01, self.cost_to_raise)
        yield f"Response from {self.config.model}"

    async def _call_with_tools_impl(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self.cost_to_raise is not None:
            raise BudgetExceededError("Too expensive", "per_call", 0.01, self.cost_to_raise)
        return {"content": f"Tool response from {self.config.model}", "tool_calls": None}


@pytest.mark.asyncio
async def test_cost_quality_router_downgrade():
    expensive = _MockLLM("gpt-4o")
    cheap = _MockLLM("gpt-4o-mini")

    router = CostQualityRouter(
        candidates=[expensive, cheap], max_cost_per_call_usd=0.005, on_exceed="downgrade"
    )

    messages = [{"role": "user", "content": "x" * 10000}]
    res = await router.generate_with_messages(messages)
    assert "gpt-4o-mini" in res

    events = router.consume_events()
    assert len(events) == 1
    assert events[0]["from_model"] == "gpt-4o"
    assert events[0]["to_model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_cost_quality_router_exhaustion():
    expensive = _MockLLM("gpt-4o")
    router = CostQualityRouter(
        candidates=[expensive], max_cost_per_call_usd=0.000001, on_exceed="downgrade"
    )

    with pytest.raises(BudgetExceededError):
        await router.generate_with_messages([{"role": "user", "content": "hello"}])


@pytest.mark.asyncio
async def test_agent_executor_surfaces_downgrade_react():
    expensive = _MockLLM("gpt-4o")
    cheap = _MockLLM("gpt-4o-mini")

    router = CostQualityRouter(
        candidates=[expensive, cheap], max_cost_per_call_usd=0.001, on_exceed="downgrade"
    )

    executor = AgentExecutor(AgentConfig(llm=router, tools=[], agent_type="react"))

    events = []
    async for event in executor.stream_steps("Explain quantum computing in 10000 words"):
        events.append(event)

    downgrade_events = [e for e in events if isinstance(e, CostDowngradeEvent)]
    assert len(downgrade_events) >= 1
    assert downgrade_events[0].from_model == "gpt-4o"
    assert downgrade_events[0].to_model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_agent_executor_surfaces_downgrade_function_calling():
    expensive = _MockLLM("gpt-4o")
    cheap = _MockLLM("gpt-4o-mini")

    router = CostQualityRouter(
        candidates=[expensive, cheap], max_cost_per_call_usd=0.001, on_exceed="downgrade"
    )

    executor = AgentExecutor(AgentConfig(llm=router, tools=[], agent_type="function_calling"))

    events = []
    async for event in executor.stream_steps("Explain quantum computing in 10000 words"):
        events.append(event)

    downgrade_events = [e for e in events if isinstance(e, CostDowngradeEvent)]
    assert len(downgrade_events) >= 1
    assert downgrade_events[0].from_model == "gpt-4o"
    assert downgrade_events[0].to_model == "gpt-4o-mini"
