from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from ..llm.base import BaseLLM
from .base import BaseTool
from .memory import AgentMemory, AgentStep
from .registry import ToolRegistry


class FunctionCallingAgent:
    """
    Agent that uses native LLM function-calling (OpenAI tool_calls / Anthropic tool_use).

    Falls back gracefully: if the LLM doesn't support call_with_tools(),
    raises RuntimeError with a suggestion to use ReActAgent instead.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool],
        max_iterations: int = 10,
        memory: AgentMemory | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._llm = llm
        self._registry = ToolRegistry(tools)
        self._max_iterations = max_iterations
        self._memory = memory or AgentMemory(max_steps=max_iterations)
        self._system_prompt = system_prompt

    def __repr__(self) -> str:
        llm = type(self._llm).__name__
        tools = len(self._registry.schemas())
        return f"FunctionCallingAgent(llm={llm!r}, tools={tools}, max_iterations={self._max_iterations})"

    def _check_support(self) -> None:
        # Check if the provider has overridden _call_with_tools_impl (not just the base NotImplementedError)
        method = getattr(type(self._llm), "_call_with_tools_impl", None)
        if method is getattr(BaseLLM, "_call_with_tools_impl", None):
            raise RuntimeError(
                f"{type(self._llm).__name__} does not support native function calling. "
                "Use ReActAgent instead, or switch to OpenAILLM / AnthropicLLM / GeminiLLM / MistralLLM."
            )

    async def run(self, query: str) -> str:
        """Run the function-calling loop and return the final answer."""
        self._check_support()
        self._memory.clear()

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]

        tool_schemas = self._registry.schemas()

        for _ in range(self._max_iterations):
            result: dict[str, Any] = await self._llm.call_with_tools(messages, tool_schemas)

            tool_calls = result.get("tool_calls")
            content = result.get("content")

            # No tool calls → final answer
            if not tool_calls:
                return content or ""

            # Append assistant message with tool_calls
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Execute each tool and append observations
            for tc in tool_calls:
                try:
                    tool = self._registry.get(tc["name"])
                    tool_result = await tool.run(**tc["arguments"])
                    observation = str(tool_result)
                except KeyError as e:
                    observation = f"Error: {e}"
                except Exception as e:
                    observation = f"Tool error: {e}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": observation,
                    }
                )

                self._memory.add_step(
                    AgentStep(
                        thought="",
                        action=tc["name"],
                        action_input=json.dumps(tc["arguments"]),
                        observation=observation,
                    )
                )

        return "I was unable to complete the task within the allowed number of steps."

    async def stream(self, query: str) -> AsyncGenerator[str]:
        """Stream the final answer (intermediate tool calls run silently)."""
        answer = await self.run(query)
        for word in answer.split(" "):
            yield word + " "

    async def stream_steps(self, query: str) -> AsyncGenerator:
        """Stream step-by-step events for function-calling agent.

        Yields ``StepEvent`` instances (``ActionEvent``, ``ObservationEvent``,
        ``TokenEvent``, ``FinalAnswerEvent``, ``ErrorEvent``).
        """
        from .step_events import (
            ActionEvent,
            ErrorEvent,
            FinalAnswerEvent,
            ObservationEvent,
            TokenEvent,
        )

        self._check_support()
        self._memory.clear()

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]

        tool_schemas = self._registry.schemas()

        for _ in range(self._max_iterations):
            result: dict[str, Any] = await self._llm.call_with_tools(messages, tool_schemas)

            tool_calls = result.get("tool_calls")
            content = result.get("content")

            # No tool calls → final answer
            if not tool_calls:
                answer = content or ""
                for token in answer.split(" "):
                    yield TokenEvent(token=token + " ")
                yield FinalAnswerEvent(answer=answer)
                return

            # Append assistant message with tool_calls
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            # Execute each tool
            for tc in tool_calls:
                yield ActionEvent(tool=tc["name"], tool_input=tc["arguments"])

                try:
                    tool = self._registry.get(tc["name"])
                    tool_result = await tool.run(**tc["arguments"])
                    observation = str(tool_result)
                except Exception as e:
                    observation = f"Error: {e}"
                    yield ErrorEvent(error=str(e))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": observation,
                    }
                )
                yield ObservationEvent(observation=observation, tool=tc["name"])

                self._memory.add_step(
                    AgentStep(
                        thought="",
                        action=tc["name"],
                        action_input=json.dumps(tc["arguments"]),
                        observation=observation,
                    )
                )

        yield FinalAnswerEvent(
            answer="I was unable to complete the task within the allowed number of steps."
        )

    @property
    def memory(self) -> AgentMemory:
        return self._memory
