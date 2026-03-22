"""Tests for stream_steps() on ReActAgent and FunctionCallingAgent (v1.3.0)."""

from __future__ import annotations

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.agents.function_calling import FunctionCallingAgent
from synapsekit.agents.react import ReActAgent
from synapsekit.agents.step_events import (
    ActionEvent,
    ErrorEvent,
    FinalAnswerEvent,
    ObservationEvent,
    ThoughtEvent,
    TokenEvent,
)
from synapsekit.llm.base import BaseLLM, LLMConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes input"

    async def run(self, *, input: str = "", **kw) -> ToolResult:
        return ToolResult(output=f"Echo: {input}")


class _StreamingMockLLM(BaseLLM):
    """Mock LLM that streams tokens and supports message-based generation."""

    def __init__(self, responses: list[str]):
        super().__init__(LLMConfig(model="mock", api_key="test", provider="openai"))
        self._responses = list(responses)
        self._call_idx = 0

    async def stream(self, prompt, **kw):
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        for char in resp:
            yield char

    async def stream_with_messages(self, messages, **kw):
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        for char in resp:
            yield char

    async def generate_with_messages(self, messages, **kw):
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return resp


class _FunctionCallingMockLLM(BaseLLM):
    """Mock LLM that supports call_with_tools for FunctionCallingAgent."""

    def __init__(self, responses: list[dict]):
        super().__init__(LLMConfig(model="mock", api_key="test", provider="openai"))
        self._responses = list(responses)
        self._call_idx = 0

    async def stream(self, prompt, **kw):
        yield "mock"

    async def _call_with_tools_impl(self, messages, tools):
        resp = self._responses[min(self._call_idx, len(self._responses) - 1)]
        self._call_idx += 1
        return resp


# ===========================================================================
# ReActAgent stream_steps tests
# ===========================================================================


class TestReActStreamSteps:
    async def test_emits_final_answer_event(self):
        llm = _StreamingMockLLM(["Thought: I know.\nFinal Answer: 42"])
        agent = ReActAgent(llm=llm, tools=[_EchoTool()])

        events = []
        async for event in agent.stream_steps("What is 6*7?"):
            events.append(event)

        # Should have TokenEvents and a FinalAnswerEvent
        token_events = [e for e in events if isinstance(e, TokenEvent)]
        final_events = [e for e in events if isinstance(e, FinalAnswerEvent)]
        assert len(token_events) > 0
        assert len(final_events) == 1
        assert final_events[0].answer == "42"

    async def test_emits_action_and_observation_events(self):
        responses = [
            "Thought: I should use echo.\nAction: echo\nAction Input: hello",
            "Thought: I now know the answer.\nFinal Answer: Echo: hello",
        ]
        llm = _StreamingMockLLM(responses)
        agent = ReActAgent(llm=llm, tools=[_EchoTool()])

        events = []
        async for event in agent.stream_steps("Echo hello"):
            events.append(event)

        action_events = [e for e in events if isinstance(e, ActionEvent)]
        obs_events = [e for e in events if isinstance(e, ObservationEvent)]
        thought_events = [e for e in events if isinstance(e, ThoughtEvent)]

        assert len(action_events) >= 1
        assert action_events[0].tool == "echo"
        assert len(obs_events) >= 1
        assert "Echo: hello" in obs_events[0].observation
        assert len(thought_events) >= 1

    async def test_step_event_types(self):
        """Verify all event type fields are correct."""
        assert ThoughtEvent(thought="t").type == "thought"
        assert ActionEvent(tool="t").type == "action"
        assert ObservationEvent(observation="o").type == "observation"
        assert TokenEvent(token="t").type == "token"
        assert FinalAnswerEvent(answer="a").type == "final_answer"
        assert ErrorEvent(error="e").type == "error"


# ===========================================================================
# FunctionCallingAgent stream_steps tests
# ===========================================================================


class _FCEchoTool(BaseTool):
    name = "echo"
    description = "Echoes input"

    async def run(self, *, text: str = "", **kw) -> ToolResult:
        return ToolResult(output=f"Echo: {text}")


class TestFunctionCallingStreamSteps:
    async def test_emits_final_answer_no_tools(self):
        llm = _FunctionCallingMockLLM(
            [
                {"content": "The answer is 42", "tool_calls": None},
            ]
        )
        agent = FunctionCallingAgent(llm=llm, tools=[_FCEchoTool()])

        events = []
        async for event in agent.stream_steps("What is 6*7?"):
            events.append(event)

        final_events = [e for e in events if isinstance(e, FinalAnswerEvent)]
        assert len(final_events) == 1
        assert final_events[0].answer == "The answer is 42"

    async def test_emits_action_and_observation(self):
        llm = _FunctionCallingMockLLM(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {"id": "tc1", "name": "echo", "arguments": {"text": "hello"}},
                    ],
                },
                {"content": "Done: Echo: hello", "tool_calls": None},
            ]
        )
        agent = FunctionCallingAgent(llm=llm, tools=[_FCEchoTool()])

        events = []
        async for event in agent.stream_steps("Echo hello"):
            events.append(event)

        action_events = [e for e in events if isinstance(e, ActionEvent)]
        obs_events = [e for e in events if isinstance(e, ObservationEvent)]

        assert len(action_events) >= 1
        assert action_events[0].tool == "echo"
        assert len(obs_events) >= 1
        assert "Echo: hello" in obs_events[0].observation

    async def test_token_events_for_final_answer(self):
        llm = _FunctionCallingMockLLM(
            [
                {"content": "hello world", "tool_calls": None},
            ]
        )
        agent = FunctionCallingAgent(llm=llm, tools=[_FCEchoTool()])

        events = []
        async for event in agent.stream_steps("hi"):
            events.append(event)

        token_events = [e for e in events if isinstance(e, TokenEvent)]
        assert len(token_events) > 0
