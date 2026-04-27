from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from ..llm.base import BaseLLM
from ..memory.agent_memory import AgentMemory as PersistentAgentMemory
from .base import BaseTool
from .memory import AgentScratchpad, AgentStep
from .registry import ToolRegistry


class FunctionCallingAgent:
    """Function-calling agent with optional persistent memory integration."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool],
        max_iterations: int = 10,
        memory: PersistentAgentMemory | AgentScratchpad | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        *,
        agent_id: str = "default",
        memory_top_k: int = 5,
        scratchpad: AgentScratchpad | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._llm = llm
        self._registry = ToolRegistry(tools)
        self._max_iterations = max_iterations
        self._system_prompt = system_prompt
        self._agent_id = agent_id
        self._memory_top_k = memory_top_k

        self._persistent_memory: PersistentAgentMemory | None = None
        if isinstance(memory, AgentScratchpad):
            self._scratchpad = memory
        else:
            self._scratchpad = scratchpad or AgentScratchpad(max_steps=max_iterations)
            if memory is not None:
                self._persistent_memory = memory

    def __repr__(self) -> str:
        llm = type(self._llm).__name__
        tools = len(self._registry.schemas())
        return f"FunctionCallingAgent(llm={llm!r}, tools={tools}, max_iterations={self._max_iterations})"

    def _check_support(self) -> None:
        method = getattr(type(self._llm), "_call_with_tools_impl", None)
        if method is getattr(BaseLLM, "_call_with_tools_impl", None):
            raise RuntimeError(
                f"{type(self._llm).__name__} does not support native function calling. "
                "Use ReActAgent instead, or switch to OpenAILLM / AnthropicLLM / GeminiLLM / MistralLLM."
            )

    async def _build_system_prompt(self, query: str) -> str:
        if self._persistent_memory is None:
            return self._system_prompt
        records = await self._persistent_memory.recall(
            agent_id=self._agent_id,
            query=query,
            top_k=self._memory_top_k,
        )
        if not records:
            return self._system_prompt

        lines = []
        for rec in records:
            memory_type = getattr(rec, "memory_type", "semantic")
            content = getattr(rec, "content", "")
            if content:
                lines.append(f"- [{memory_type}] {content}")
        if not lines:
            return self._system_prompt

        return (
            f"{self._system_prompt}\n\n"
            "Relevant persistent memories (use when helpful and factual):\n" + "\n".join(lines)
        )

    async def _store_episode(self, query: str, answer: str) -> None:
        if self._persistent_memory is None:
            return
        await self._persistent_memory.store(
            agent_id=self._agent_id,
            content=f"User query: {query}\nAgent answer: {answer}",
            memory_type="episodic",
            metadata={
                "tool_count": len(self._scratchpad.steps),
                "tools": [step.action for step in self._scratchpad.steps],
            },
        )

    async def run(self, query: str) -> str:
        self._check_support()
        self._scratchpad.clear()

        messages: list[dict] = [
            {"role": "system", "content": await self._build_system_prompt(query)},
            {"role": "user", "content": query},
        ]

        tool_schemas = self._registry.schemas()

        for _ in range(self._max_iterations):
            result: dict[str, Any] = await self._llm.call_with_tools(messages, tool_schemas)

            tool_calls = result.get("tool_calls")
            content = result.get("content")

            if not tool_calls:
                final_answer = content or ""
                await self._store_episode(query, final_answer)
                return final_answer

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

                self._scratchpad.add_step(
                    AgentStep(
                        thought="",
                        action=tc["name"],
                        action_input=json.dumps(tc["arguments"]),
                        observation=observation,
                    )
                )

        fallback = "I was unable to complete the task within the allowed number of steps."
        await self._store_episode(query, fallback)
        return fallback

    async def stream(self, query: str) -> AsyncGenerator[str]:
        answer = await self.run(query)
        for word in answer.split(" "):
            yield word + " "

    async def stream_steps(self, query: str) -> AsyncGenerator:
        """Stream step-by-step events for function-calling agent."""
        from .step_events import (
            ActionEvent,
            CostDowngradeEvent,
            ErrorEvent,
            FinalAnswerEvent,
            ObservationEvent,
            TokenEvent,
        )

        self._check_support()
        self._scratchpad.clear()

        messages: list[dict] = [
            {"role": "system", "content": await self._build_system_prompt(query)},
            {"role": "user", "content": query},
        ]

        tool_schemas = self._registry.schemas()

        for _ in range(self._max_iterations):
            # Check for cost-downgrade events from router (pre-call)
            if hasattr(self._llm, "consume_events"):
                for raw_event in self._llm.consume_events():
                    yield CostDowngradeEvent(
                        from_model=raw_event["from_model"],
                        to_model=raw_event["to_model"],
                        reason=raw_event["reason"],
                    )

            result: dict[str, Any] = await self._llm.call_with_tools(messages, tool_schemas)

            # Check for cost-downgrade events after call
            if hasattr(self._llm, "consume_events"):
                for raw_event in self._llm.consume_events():
                    yield CostDowngradeEvent(
                        from_model=raw_event["from_model"],
                        to_model=raw_event["to_model"],
                        reason=raw_event["reason"],
                    )

            tool_calls = result.get("tool_calls")
            content = result.get("content")

            if not tool_calls:
                answer = content or ""
                await self._store_episode(query, answer)
                for token in answer.split(" "):
                    yield TokenEvent(token=token + " ")
                yield FinalAnswerEvent(answer=answer)
                return

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

                self._scratchpad.add_step(
                    AgentStep(
                        thought="",
                        action=tc["name"],
                        action_input=json.dumps(tc["arguments"]),
                        observation=observation,
                    )
                )

        fallback = "I was unable to complete the task within the allowed number of steps."
        await self._store_episode(query, fallback)
        yield FinalAnswerEvent(answer=fallback)

    @property
    def memory(self) -> AgentScratchpad:
        return self._scratchpad

    @property
    def persistent_memory(self) -> PersistentAgentMemory | None:
        return self._persistent_memory
