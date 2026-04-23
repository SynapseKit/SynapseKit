"""Production-grade tests for SimpleAgent and agent() factory."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.agents.facade import SimpleAgent, agent
from synapsekit.memory.conversation import ConversationMemory

# ── SimpleAgent structure ────────────────────────────────────────────────────


class TestSimpleAgentStructure:
    def test_arun_is_coroutine(self):
        executor = MagicMock()
        sa = SimpleAgent(executor=executor)
        assert inspect.iscoroutinefunction(sa.arun)

    def test_run_is_sync(self):
        executor = MagicMock()
        sa = SimpleAgent(executor=executor)
        assert not inspect.iscoroutinefunction(sa.run)

    def test_no_memory_by_default(self):
        executor = MagicMock()
        sa = SimpleAgent(executor=executor)
        assert sa._memory is None


# ── SimpleAgent.arun ────────────────────────────────────────────────────────


class TestSimpleAgentArun:
    @pytest.mark.asyncio
    async def test_arun_calls_executor(self):
        executor = MagicMock()
        executor.run = AsyncMock(return_value="hello")
        sa = SimpleAgent(executor=executor)
        result = await sa.arun("hi")
        assert result == "hello"
        executor.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_arun_returns_string(self):
        executor = MagicMock()
        executor.run = AsyncMock(return_value="answer")
        sa = SimpleAgent(executor=executor)
        result = await sa.arun("question")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_arun_updates_memory(self):
        executor = MagicMock()
        executor.run = AsyncMock(return_value="the sky is blue")
        memory = ConversationMemory()
        sa = SimpleAgent(executor=executor, memory=memory)
        await sa.arun("what color is the sky?")
        assert len(memory) == 2  # user + assistant turn


# ── SimpleAgent.run (sync) ───────────────────────────────────────────────────


class TestSimpleAgentRun:
    def test_run_calls_executor_sync(self):
        executor = MagicMock()
        executor.run_sync = MagicMock(return_value="sync answer")
        sa = SimpleAgent(executor=executor)
        result = sa.run("hello")
        assert result == "sync answer"
        executor.run_sync.assert_called_once()


# ── memory context ───────────────────────────────────────────────────────────


class TestSimpleAgentMemory:
    @pytest.mark.asyncio
    async def test_memory_prepends_context_on_second_turn(self):
        captured = []

        async def fake_run(prompt: str) -> str:
            captured.append(prompt)
            return "ok"

        executor = MagicMock()
        executor.run = AsyncMock(side_effect=fake_run)
        memory = ConversationMemory()
        sa = SimpleAgent(executor=executor, memory=memory)

        await sa.arun("first message")
        await sa.arun("second message")

        # Second call's prompt should contain previous context
        assert len(captured) == 2
        assert "first message" in captured[1] or "User" in captured[1]

    @pytest.mark.asyncio
    async def test_no_memory_passes_prompt_unchanged(self):
        captured = []

        async def fake_run(prompt: str) -> str:
            captured.append(prompt)
            return "ok"

        executor = MagicMock()
        executor.run = AsyncMock(side_effect=fake_run)
        sa = SimpleAgent(executor=executor)  # no memory

        await sa.arun("plain prompt")
        assert captured[0] == "plain prompt"


# ── agent() factory ──────────────────────────────────────────────────────────


class TestAgentFactory:
    def _make_agent(self, **kwargs):
        with patch("synapsekit.agents.facade.make_llm") as mock_llm, \
             patch("synapsekit.agents.facade.AgentExecutor") as mock_exec:
            mock_llm.return_value = MagicMock()
            mock_exec.return_value = MagicMock()
            mock_exec.return_value.run = AsyncMock(return_value="ok")
            return agent(**kwargs), mock_llm, mock_exec

    def test_returns_simple_agent(self):
        sa, _, _ = self._make_agent(model="gpt-4o-mini")
        assert isinstance(sa, SimpleAgent)

    def test_api_key_defaults_to_empty_string(self):
        """api_key must have a default so local models (Ollama) don't require it."""
        _sa, mock_llm, _ = self._make_agent(model="llama3.2")
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["api_key"] == ""

    def test_explicit_api_key_is_forwarded(self):
        _sa, mock_llm, _ = self._make_agent(model="gpt-4o-mini", api_key="sk-test")
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"

    def test_memory_false_gives_no_memory(self):
        sa, _, _ = self._make_agent(model="gpt-4o-mini", memory=False)
        assert sa._memory is None

    def test_memory_true_gives_conversation_memory(self):
        sa, _, _ = self._make_agent(model="gpt-4o-mini", memory=True)
        assert isinstance(sa._memory, ConversationMemory)

    def test_system_prompt_forwarded_to_make_llm(self):
        _sa, mock_llm, _ = self._make_agent(
            model="gpt-4o-mini", system_prompt="You are a pirate."
        )
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a pirate."

    def test_custom_tools_forwarded_to_config(self):
        from synapsekit.agents.tools.calculator import CalculatorTool
        tools = [CalculatorTool()]
        with patch("synapsekit.agents.facade.make_llm") as mock_llm, \
             patch("synapsekit.agents.facade.AgentConfig") as mock_config, \
             patch("synapsekit.agents.facade.AgentExecutor"):
            mock_llm.return_value = MagicMock()
            agent(model="gpt-4o-mini", tools=tools)
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["tools"] == tools

    def test_top_level_export(self):
        from synapsekit import SimpleAgent as TopLevelSimpleAgent
        from synapsekit import agent as top_level_agent
        assert TopLevelSimpleAgent is SimpleAgent
        assert top_level_agent is agent
