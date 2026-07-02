from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass

import pytest

from synapsekit.agents import (
    AgentConfig,
    AgentConfigPatch,
    AgentEvolutionAuditLog,
    AgentExecutor,
    BaseTool,
    MetaAnalyzer,
    SelfImprovingAgent,
    ToolResult,
)
from synapsekit.evaluation import EvalSuite
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.training import FeedbackCollector, InMemoryFeedbackBackend, RolloutPolicy


class StaticLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(LLMConfig(model="static", api_key="", provider="test"))

    async def stream(self, prompt: str, **kw):
        del prompt, kw
        yield "ok"


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo input"
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}}

    async def run(self, **kwargs):
        return ToolResult(output=str(kwargs.get("input", "")))


@dataclass
class StubAnalyzer:
    patches: list[AgentConfigPatch]

    async def propose(self, *, samples, snapshot, improvement_targets):
        del samples, snapshot, improvement_targets
        return list(self.patches)


def _executor(prompt: str = "Base prompt") -> AgentExecutor:
    return AgentExecutor(
        AgentConfig(
            llm=StaticLLM(),
            tools=[EchoTool()],
            system_prompt=prompt,
            agent_type="function_calling",
        )
    )


def _suite(score_map: dict[str, float], threshold: float = 0.85) -> EvalSuite:
    async def case(prompt: str = ""):
        return {"score": score_map.get(prompt, 0.0), "cost_usd": 0.0}

    return EvalSuite.from_cases([("case", case)], threshold=threshold)


def test_self_improving_agent_applies_eval_approved_prompt_patch():
    executor = _executor()
    patch = AgentConfigPatch(
        patch_type="prompt_rewrite",
        description="Use better prompt",
        changes={"system_prompt": "Better prompt"},
    )
    agent = SelfImprovingAgent(
        executor,
        eval_suite=_suite({"Base prompt": 0.8, "Better prompt": 0.95}),
        rollout=RolloutPolicy(canary_pct=5.0, min_eval_score=0.85),
        meta_analyzer=StubAnalyzer([patch]),
    )

    result = asyncio.run(agent.evolve())

    assert result.status == "canary"
    assert executor.config.system_prompt == "Better prompt"
    assert result.patch is not None
    assert result.patch.eval_score == pytest.approx(0.95)
    assert result.patch.verify()
    assert agent.rollout_manager.state.current_pct == 5.0


def test_self_improving_agent_blocks_patch_when_eval_regresses():
    executor = _executor()
    patch = AgentConfigPatch(
        patch_type="prompt_rewrite",
        description="Worse prompt",
        changes={"system_prompt": "Worse prompt"},
    )
    agent = SelfImprovingAgent(
        executor,
        eval_suite=_suite({"Base prompt": 0.9, "Worse prompt": 0.88}, threshold=0.85),
        rollout=RolloutPolicy(min_eval_score=0.85, rollback_on_regression=True),
        meta_analyzer=StubAnalyzer([patch]),
    )

    result = asyncio.run(agent.evolve())

    assert result.status == "blocked"
    assert executor.config.system_prompt == "Base prompt"
    history = agent.evolution_history()
    assert history[0].status == "blocked"
    assert history[0].metadata["block_reason"] == "eval score regressed from 0.900 to 0.880"


def test_tool_addition_requires_human_approval_before_apply():
    executor = _executor()
    patch = AgentConfigPatch(
        patch_type="tool_addition",
        description="Add search",
        changes={"tool_name": "search"},
    )
    agent = SelfImprovingAgent(
        executor,
        eval_suite=_suite({"Base prompt": 0.9}, threshold=0.85),
        rollout=RolloutPolicy(min_eval_score=0.85),
        meta_analyzer=StubAnalyzer([patch]),
    )

    result = asyncio.run(agent.evolve())

    assert result.status == "approved"
    assert result.reason == "human approval required"
    assert executor.config.tools[0].name == "echo"


def test_approve_applies_waiting_patch_and_records_history():
    executor = _executor()
    patch = AgentConfigPatch(
        patch_type="prompt_rewrite",
        description="Needs approval",
        changes={"system_prompt": "Approved prompt"},
    )
    agent = SelfImprovingAgent(
        executor,
        eval_suite=_suite({"Base prompt": 0.8, "Approved prompt": 0.95}),
        rollout=RolloutPolicy(min_eval_score=0.85),
        meta_analyzer=StubAnalyzer([patch]),
    )

    result = asyncio.run(agent.evolve(require_approval=True))
    approved = agent.approve(result.patch.patch_id, approved_by="reviewer")

    assert approved.status == "canary"
    assert approved.approved_by == "reviewer"
    assert executor.config.system_prompt == "Approved prompt"


def test_rollback_restores_previous_snapshot():
    executor = _executor()
    patch = AgentConfigPatch(
        patch_type="prompt_rewrite",
        description="Apply then rollback",
        changes={"system_prompt": "Candidate prompt"},
    )
    agent = SelfImprovingAgent(
        executor,
        eval_suite=_suite({"Base prompt": 0.8, "Candidate prompt": 0.95}),
        rollout=RolloutPolicy(min_eval_score=0.85),
        meta_analyzer=StubAnalyzer([patch]),
    )

    result = asyncio.run(agent.evolve())
    rollback = agent.rollback(result.patch.patch_id, reason="live KPI regression")

    assert rollback.status == "rolled_back"
    assert rollback.rollback_of == result.patch.patch_id
    assert executor.config.system_prompt == "Base prompt"
    assert agent.rollout_manager.state.rolled_back is True


def test_agent_config_patch_signature_detects_tampering():
    patch = AgentConfigPatch(
        patch_type="prompt_rewrite",
        description="Signed",
        changes={"system_prompt": "x"},
    )

    patch.sign("secret")
    assert patch.verify("secret") is True
    patch.changes["system_prompt"] = "tampered"
    assert patch.verify("secret") is False


def test_agent_evolution_audit_log_round_trips_jsonl(tmp_path):
    path = tmp_path / "evolution.jsonl"
    log = AgentEvolutionAuditLog(path)
    patch = AgentConfigPatch(
        patch_type="prompt_rewrite",
        description="Persisted",
        changes={"system_prompt": "x"},
    )

    log.append(patch)
    loaded = AgentEvolutionAuditLog(path)

    assert loaded.get(patch.patch_id) is not None
    assert loaded.list()[0].description == "Persisted"


def test_meta_analyzer_heuristic_uses_negative_feedback():
    backend = InMemoryFeedbackBackend()
    collector = FeedbackCollector(backend=backend)
    sample = collector.record(
        "ambiguous question",
        "wrong answer",
        "negative",
        corrected_response="ask a clarification question",
    )
    asyncio.run(backend.save(sample))
    analyzer = MetaAnalyzer()

    async def run_propose():
        return await analyzer.propose(
            samples=await collector.get_samples(),
            snapshot=agent_snapshot("Base prompt"),
            improvement_targets=["prompt"],
        )

    patches = asyncio.run(run_propose())

    assert patches
    assert patches[0].patch_type == "prompt_rewrite"
    assert "ambiguous question" in patches[0].changes["system_prompt"]


def agent_snapshot(prompt: str):
    from synapsekit.agents.self_improving import AgentConfigSnapshot

    return AgentConfigSnapshot(system_prompt=prompt)


# ---------------------------------------------------------------------------
# Coroutine assertions
# ---------------------------------------------------------------------------


def test_self_improving_agent_async_methods_are_coroutines():
    assert inspect.iscoroutinefunction(SelfImprovingAgent.arun)
    assert inspect.iscoroutinefunction(SelfImprovingAgent.evolve)


# ---------------------------------------------------------------------------
# Negative tests for approve / rollback
# ---------------------------------------------------------------------------


def test_approve_unknown_patch_id_raises():
    agent = SelfImprovingAgent(
        _executor(),
        eval_suite=_suite({"Base prompt": 0.9}),
    )
    with pytest.raises(KeyError):
        agent.approve("nonexistent-id")


def test_rollback_unknown_patch_id_raises():
    agent = SelfImprovingAgent(
        _executor(),
        eval_suite=_suite({"Base prompt": 0.9}),
    )
    with pytest.raises(KeyError):
        agent.rollback("nonexistent-id")


# ---------------------------------------------------------------------------
# _cadence_due branch coverage
# ---------------------------------------------------------------------------


def test_cadence_manual_never_triggers_evolve():
    """cadence='manual' must never auto-trigger evolve regardless of run count."""
    agent = SelfImprovingAgent(
        _executor(),
        eval_suite=_suite({"Base prompt": 0.9}),
        cadence="manual",
    )
    # _cadence_due is private; we exercise the branch via the public interface
    for _ in range(10):
        agent._run_count += 1
    assert agent._cadence_due() is False


def test_cadence_integer_triggers_every_n_runs():
    """cadence=3 must become due exactly when _run_count is a multiple of 3."""
    agent = SelfImprovingAgent(
        _executor(),
        eval_suite=_suite({"Base prompt": 0.9}),
        cadence=3,
    )
    agent._run_count = 3
    assert agent._cadence_due() is True
    agent._run_count = 4
    assert agent._cadence_due() is False
    agent._run_count = 6
    assert agent._cadence_due() is True


# ---------------------------------------------------------------------------
# Stress test for AgentEvolutionAuditLog
# ---------------------------------------------------------------------------


def test_audit_log_handles_50_entries(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AgentEvolutionAuditLog(path)

    for i in range(50):
        patch = AgentConfigPatch(
            patch_type="prompt_rewrite",
            description=f"patch-{i}",
            changes={"system_prompt": f"prompt-{i}"},
        )
        log.append(patch)

    # Verify all 50 are in memory
    entries = log.list()
    assert len(entries) == 50

    # Verify round-trip via disk
    reloaded = AgentEvolutionAuditLog(path)
    assert len(reloaded.list()) == 50
    assert isinstance(reloaded.list()[0], AgentConfigPatch)
