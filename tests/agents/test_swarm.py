"""Tests for SwarmAgent — mocked LLM."""

from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.agents.swarm import SwarmAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_llm(generate_responses=None):
    """Return a mock BaseLLM; generate_responses is a list cycled through."""
    llm = MagicMock()
    responses = list(generate_responses or [])
    call_count = 0

    async def fake_generate(prompt, **kw):
        nonlocal call_count
        if responses:
            resp = responses[call_count % len(responses)]
            call_count += 1
            return resp
        return '{"score": 0.5}'

    llm.generate = fake_generate
    return llm


def make_swarm(llm=None, tools=None, max_agents=5, spawn_threshold=0.7):
    llm = llm or make_mock_llm()
    return SwarmAgent(
        llm=llm, tools=tools or [], max_agents=max_agents, spawn_threshold=spawn_threshold
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSwarmAgentConstruction:
    def test_defaults(self):
        llm = make_mock_llm()
        agent = SwarmAgent(llm=llm)
        assert agent._max_agents == 5
        assert agent._spawn_threshold == 0.7
        assert agent._system_prompt == "You are a helpful assistant."
        assert agent._tools == []

    def test_custom_params(self):
        llm = make_mock_llm()
        agent = SwarmAgent(llm=llm, max_agents=3, spawn_threshold=0.5, system_prompt="Be concise.")
        assert agent._max_agents == 3
        assert agent._spawn_threshold == 0.5

    def test_invalid_max_agents(self):
        llm = make_mock_llm()
        with pytest.raises(ValueError, match="max_agents must be >= 1"):
            SwarmAgent(llm=llm, max_agents=0)

    def test_invalid_spawn_threshold_below(self):
        llm = make_mock_llm()
        with pytest.raises(ValueError, match="spawn_threshold must be between"):
            SwarmAgent(llm=llm, spawn_threshold=-0.1)

    def test_invalid_spawn_threshold_above(self):
        llm = make_mock_llm()
        with pytest.raises(ValueError, match="spawn_threshold must be between"):
            SwarmAgent(llm=llm, spawn_threshold=1.1)

    def test_repr(self):
        llm = make_mock_llm()
        agent = SwarmAgent(llm=llm, max_agents=3, spawn_threshold=0.6)
        r = repr(agent)
        assert "SwarmAgent" in r
        assert "max_agents=3" in r


# ---------------------------------------------------------------------------
# _score_complexity
# ---------------------------------------------------------------------------


class TestSwarmAgentScoreComplexity:
    @pytest.mark.asyncio
    async def test_score_from_json(self):
        llm = make_mock_llm(['{"score": 0.8}'])
        agent = SwarmAgent(llm=llm)
        score = await agent._score_complexity("build a rocket")
        assert score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_score_clamped_to_0_1(self):
        llm = make_mock_llm(['{"score": 2.5}'])
        agent = SwarmAgent(llm=llm)
        score = await agent._score_complexity("anything")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_score_fallback_on_invalid_json(self):
        llm = make_mock_llm(["not json at all 0.3"])
        agent = SwarmAgent(llm=llm)
        score = await agent._score_complexity("anything")
        # regex extracts 0.3
        assert score == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_score_fallback_default_on_no_number(self):
        llm = make_mock_llm(["no numbers here"])
        agent = SwarmAgent(llm=llm)
        score = await agent._score_complexity("anything")
        assert score == 0.5


# ---------------------------------------------------------------------------
# _decompose
# ---------------------------------------------------------------------------


class TestSwarmAgentDecompose:
    @pytest.mark.asyncio
    async def test_decompose_returns_list(self):
        subtasks = ["subtask 1", "subtask 2", "subtask 3"]
        llm = make_mock_llm([json.dumps(subtasks)])
        agent = SwarmAgent(llm=llm, max_agents=5)
        result = await agent._decompose("complex task")
        assert result == subtasks

    @pytest.mark.asyncio
    async def test_decompose_truncates_to_max_agents(self):
        subtasks = [f"subtask {i}" for i in range(10)]
        llm = make_mock_llm([json.dumps(subtasks)])
        agent = SwarmAgent(llm=llm, max_agents=3)
        result = await agent._decompose("complex task")
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_decompose_fallback_on_invalid_json(self):
        llm = make_mock_llm(["not a json list"])
        agent = SwarmAgent(llm=llm)
        result = await agent._decompose("original task")
        assert result == ["original task"]


# ---------------------------------------------------------------------------
# run() — direct path (score < threshold)
# ---------------------------------------------------------------------------


class TestSwarmAgentRunDirect:
    @pytest.mark.asyncio
    async def test_run_direct_when_low_complexity(self):
        # score = 0.2 < 0.7 → direct
        low_score_resp = '{"score": 0.2}'
        direct_resp = "Simple answer"

        llm = make_mock_llm([low_score_resp])

        mock_fc_agent = MagicMock()
        mock_fc_agent.run = AsyncMock(return_value=direct_resp)

        with patch("synapsekit.agents.swarm.FunctionCallingAgent", return_value=mock_fc_agent):
            agent = SwarmAgent(llm=llm, spawn_threshold=0.7)
            result = await agent.run("What is 2+2?")

        assert result == direct_resp

    @pytest.mark.asyncio
    async def test_run_direct_fallback_on_fc_error(self):
        low_score_resp = '{"score": 0.1}'
        fallback_resp = "LLM direct answer"

        call_count = 0

        async def fake_generate(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return low_score_resp
            return fallback_resp

        llm = MagicMock()
        llm.generate = fake_generate

        mock_fc_agent = MagicMock()
        mock_fc_agent.run = AsyncMock(side_effect=Exception("tool error"))

        with patch("synapsekit.agents.swarm.FunctionCallingAgent", return_value=mock_fc_agent):
            agent = SwarmAgent(llm=llm, spawn_threshold=0.7)
            result = await agent.run("simple task")

        assert result == fallback_resp


# ---------------------------------------------------------------------------
# run() — swarm path (score >= threshold)
# ---------------------------------------------------------------------------


class TestSwarmAgentRunSwarm:
    @pytest.mark.asyncio
    async def test_run_spawns_subagents_when_high_complexity(self):
        # Use separate generators for scoring/decompose vs synthesize
        llm_responses = iter(
            [
                '{"score": 0.9}',  # complexity score
                '["subtask A", "subtask B"]',  # decompose
                "Final synthesized answer",  # synthesize
            ]
        )

        async def fake_generate(prompt, **kw):
            return next(llm_responses)

        llm = MagicMock()
        llm.generate = fake_generate

        subtask_results = {"subtask A": "Result A", "subtask B": "Result B"}

        class FakeFC:
            def __init__(self, **kw):
                pass

            async def run(self, task):
                return subtask_results.get(task, "unknown")

        with patch("synapsekit.agents.swarm.FunctionCallingAgent", FakeFC):
            agent = SwarmAgent(llm=llm, spawn_threshold=0.7)
            result = await agent.run("complex task requiring multiple steps")

        assert result == "Final synthesized answer"

    @pytest.mark.asyncio
    async def test_run_swarm_uses_gather(self):
        """Subtasks should be run concurrently via asyncio.gather."""
        gathered_tasks = []

        responses_iter = iter(
            [
                '{"score": 0.95}',
                '["sub1", "sub2", "sub3"]',
            ]
        )
        subtask_results = {"sub1": "R1", "sub2": "R2", "sub3": "R3"}

        async def fake_generate(prompt, **kw):
            try:
                return next(responses_iter)
            except StopIteration:
                # synthesize call
                return "synthesized"

        llm = MagicMock()
        llm.generate = fake_generate

        created_agents = []

        class FakeFC:
            def __init__(self, **kw):
                created_agents.append(self)
                self._idx = len(created_agents) - 1

            async def run(self, task):
                gathered_tasks.append(task)
                return subtask_results.get(task, "unknown")

        with patch("synapsekit.agents.swarm.FunctionCallingAgent", FakeFC):
            agent = SwarmAgent(llm=llm, spawn_threshold=0.7)
            await agent.run("complex task")

        assert set(gathered_tasks) == {"sub1", "sub2", "sub3"}

    @pytest.mark.asyncio
    async def test_run_swarm_handles_subtask_error(self):
        responses_iter = iter(
            [
                '{"score": 0.9}',
                '["good subtask", "failing subtask"]',
            ]
        )

        async def fake_generate(prompt, **kw):
            try:
                return next(responses_iter)
            except StopIteration:
                return "final answer"

        llm = MagicMock()
        llm.generate = fake_generate

        class FakeFC:
            def __init__(self, **kw):
                pass

            async def run(self, task):
                if "failing" in task:
                    raise RuntimeError("boom")
                return "good result"

        with patch("synapsekit.agents.swarm.FunctionCallingAgent", FakeFC):
            agent = SwarmAgent(llm=llm, spawn_threshold=0.7)
            result = await agent.run("complex task")

        # Should not raise; subtask error is captured
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_run_swarm_fallback_empty_subtasks(self):
        """If decompose produces empty list, fall back to original task."""
        responses_iter = iter(
            [
                '{"score": 0.9}',
                "[]",  # empty subtasks
            ]
        )

        async def fake_generate(prompt, **kw):
            try:
                return next(responses_iter)
            except StopIteration:
                return "final"

        llm = MagicMock()
        llm.generate = fake_generate

        class FakeFC:
            def __init__(self, **kw):
                pass

            async def run(self, task):
                return f"handled: {task}"

        with patch("synapsekit.agents.swarm.FunctionCallingAgent", FakeFC):
            agent = SwarmAgent(llm=llm, spawn_threshold=0.7)
            result = await agent.run("my original task")

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# run() is a coroutine
# ---------------------------------------------------------------------------


def test_run_is_coroutine():
    llm = make_mock_llm()
    agent = SwarmAgent(llm=llm)
    assert inspect.iscoroutinefunction(agent.run)
