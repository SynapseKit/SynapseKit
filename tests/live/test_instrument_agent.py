"""Self-evolving agent activity streams to Live.

Real objects only — a scripted BaseLLM, a real EvalSuite/RolloutPolicy and a
deterministic analyzer. No server: instrument the classes, toggle the bus, and
read what was published.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from synapsekit import (
    AgentConfig,
    AgentConfigPatch,
    AgentExecutor,
    BaseLLM,
    EvalSuite,
    LLMConfig,
    RolloutPolicy,
    SelfImprovingAgent,
)
from synapsekit.live import bus
from synapsekit.live.instrument import instrument_all


class _ScriptedLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig(provider="demo", model="scripted", api_key=""))

    async def stream(self, prompt: str, **kw: object):
        del prompt, kw
        yield "ok"


class _Analyzer:
    async def propose(self, *, samples, snapshot, improvement_targets):
        del samples, snapshot, improvement_targets
        return [
            AgentConfigPatch(
                patch_type="prompt_rewrite",
                description="Prefer clarification.",
                changes={"system_prompt": "Ask a clarification question when ambiguous."},
                metadata={"directive": "refuse_ambiguous"},
            )
        ]


async def _eval_case(prompt: str = "") -> dict[str, float]:
    # Any prompt carrying the proposed directive beats the bare baseline.
    return {"score": 0.95 if "clarification" in prompt else 0.70, "cost_usd": 0.0}


def _build_agent(tmp_path: Path) -> SelfImprovingAgent:
    executor = AgentExecutor(
        AgentConfig(llm=_ScriptedLLM(), tools=[], system_prompt="You are a helpful assistant.")
    )
    return SelfImprovingAgent(
        executor,
        eval_suite=EvalSuite.from_cases([("clarify", _eval_case)], threshold=None),
        rollout=RolloutPolicy(min_eval_score=0.85, require_human_approval_for=[]),
        meta_analyzer=_Analyzer(),
        improvement_targets=["prompt"],
        agent_id="live-test-agent",
        audit_path=tmp_path / "evolution.jsonl",
    )


@pytest.fixture(autouse=True)
def _instrumented_bus():
    instrument_all()
    was = bus.enabled
    bus.enabled = True
    bus.clear()
    yield
    bus.enabled = was


def test_evolve_stays_a_coroutine_after_instrumentation() -> None:
    assert inspect.iscoroutinefunction(SelfImprovingAgent.evolve)


async def test_evolution_cycle_publishes_agent_evolve(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    outcome = await agent.evolve()
    assert outcome.patch is not None  # sanity: the patch was accepted

    events = [e for e in bus.history() if e["kind"] == "agent.evolve"]
    assert events, "agent.evolve not published"
    attrs = events[-1]["attributes"]
    assert attrs["agent_id"] == "live-test-agent"
    assert attrs["patch_status"] in {"canary", "promoted"}
    assert attrs["directive"] == "refuse_ambiguous"
    assert events[-1]["status"] == "ok"


def test_rollback_publishes_agent_rollback(tmp_path: Path) -> None:
    import asyncio

    agent = _build_agent(tmp_path)
    patch = asyncio.run(agent.evolve()).patch
    assert patch is not None
    bus.clear()

    agent.rollback(patch.patch_id, reason="live probe")
    events = [e for e in bus.history() if e["kind"] == "agent.rollback"]
    assert events, "agent.rollback not published"
    attrs = events[-1]["attributes"]
    assert attrs["agent_id"] == "live-test-agent"
    assert attrs["rolled_back"] == patch.patch_id[:8]
    assert attrs["reason"] == "live probe"
