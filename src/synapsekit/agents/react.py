from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any

from ..llm.base import BaseLLM
from ..memory.agent_memory import AgentMemory as PersistentAgentMemory
from .base import BaseTool
from .memory import AgentScratchpad, AgentStep
from .registry import ToolRegistry

_REACT_SYSTEM = """\
You are a helpful AI assistant with access to tools.

Available tools:
{tools}

Use EXACTLY this format for every response until you have a final answer:

Thought: (your reasoning about what to do next)
Action: (the exact tool name from the list above)
Action Input: (the input to pass to the tool, as a plain string)

When you have enough information to answer:

Thought: I now know the final answer.
Final Answer: (your complete answer to the original question)

Rules:
- Only use tools from the list above.
- Never invent tool results — always call the tool and wait for the Observation.
- Never skip the Thought step.
- Provide Final Answer only when you are confident.
"""

_ACTION_RE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(
    r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|$)", re.IGNORECASE | re.DOTALL
)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip() if m else ""


def _parse_action(text: str) -> tuple[str, str]:
    action_m = _ACTION_RE.search(text)
    input_m = _ACTION_INPUT_RE.search(text)
    action = action_m.group(1).strip() if action_m else ""
    action_input = input_m.group(1).strip() if input_m else ""
    return action, action_input


def _parse_final_answer(text: str) -> str | None:
    m = _FINAL_ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


class ReActAgent:
    """Reasoning + Acting agent with optional persistent memory integration."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool],
        max_iterations: int = 10,
        memory: PersistentAgentMemory | AgentScratchpad | None = None,
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
        return f"ReActAgent(llm={llm!r}, tools={tools}, max_iterations={self._max_iterations})"

    @staticmethod
    def _format_recalled_memories(records: list[Any]) -> str:
        if not records:
            return ""
        lines = []
        for rec in records:
            memory_type = getattr(rec, "memory_type", "semantic")
            content = getattr(rec, "content", "")
            if content:
                lines.append(f"- [{memory_type}] {content}")
        return "\n".join(lines)

    async def _recall_context(self, query: str) -> str:
        if self._persistent_memory is None:
            return ""
        records = await self._persistent_memory.recall(
            agent_id=self._agent_id,
            query=query,
            top_k=self._memory_top_k,
        )
        return self._format_recalled_memories(records)

    async def _store_episode(self, query: str, answer: str) -> None:
        if self._persistent_memory is None:
            return
        tool_names = [step.action for step in self._scratchpad.steps]
        await self._persistent_memory.store(
            agent_id=self._agent_id,
            content=f"User query: {query}\nAgent answer: {answer}",
            memory_type="episodic",
            metadata={
                "tool_count": len(tool_names),
                "tools": tool_names,
            },
        )

    def _build_system_prompt(self, memory_context: str = "") -> str:
        prompt = _REACT_SYSTEM.format(tools=self._registry.describe())
        if not memory_context:
            return prompt
        return (
            f"{prompt}\n\n"
            "Relevant persistent memories (use when helpful and factual):\n"
            f"{memory_context}"
        )

    def _build_messages(self, query: str, memory_context: str = "") -> list[dict]:
        scratchpad = self._scratchpad.format_scratchpad()
        user_content = f"Question: {query}"
        if scratchpad:
            user_content += f"\n\n{scratchpad}"
        return [
            {"role": "system", "content": self._build_system_prompt(memory_context)},
            {"role": "user", "content": user_content},
        ]

    async def run(self, query: str) -> str:
        """Run the ReAct loop and return the final answer."""
        self._scratchpad.clear()
        memory_context = await self._recall_context(query)

        for _ in range(self._max_iterations):
            messages = self._build_messages(query, memory_context)
            response = await self._llm.generate_with_messages(messages)

            final = _parse_final_answer(response)
            if final is not None:
                await self._store_episode(query, final)
                return final

            action_name, action_input = _parse_action(response)
            thought = _parse_thought(response)

            if not action_name:
                final_answer = response.strip()
                await self._store_episode(query, final_answer)
                return final_answer

            try:
                tool = self._registry.get(action_name)
                result = await tool.run(input=action_input)
                observation = str(result)
            except KeyError as e:
                observation = f"Error: {e}"
            except Exception as e:
                observation = f"Tool error: {e}"

            self._scratchpad.add_step(
                AgentStep(
                    thought=thought,
                    action=action_name,
                    action_input=action_input,
                    observation=observation,
                )
            )

        fallback = "I was unable to find the answer within the allowed number of steps."
        await self._store_episode(query, fallback)
        return fallback

    async def stream(self, query: str) -> AsyncGenerator[str]:
        answer = await self.run(query)
        for word in answer.split(" "):
            yield word + " "

    async def stream_steps(self, query: str) -> AsyncGenerator:
        """Stream step-by-step events including thoughts, actions, and tokens."""
        from .step_events import (
            ActionEvent,
            CostDowngradeEvent,
            ErrorEvent,
            FinalAnswerEvent,
            ObservationEvent,
            ThoughtEvent,
            TokenEvent,
        )

        self._scratchpad.clear()
        memory_context = await self._recall_context(query)

        for _ in range(self._max_iterations):
            messages = self._build_messages(query, memory_context)

            # Check for cost-downgrade events from router (pre-call)
            if hasattr(self._llm, "consume_events"):
                for raw_event in self._llm.consume_events():
                    yield CostDowngradeEvent(
                        from_model=raw_event["from_model"],
                        to_model=raw_event["to_model"],
                        reason=raw_event["reason"],
                    )

            full_response = ""
            async for token in self._llm.stream_with_messages(messages):
                # Also check during stream if supported (rare)
                if hasattr(self._llm, "consume_events"):
                    for raw_event in self._llm.consume_events():
                        yield CostDowngradeEvent(
                            from_model=raw_event["from_model"],
                            to_model=raw_event["to_model"],
                            reason=raw_event["reason"],
                        )
                yield TokenEvent(token=token)
                full_response += token

            final = _parse_final_answer(full_response)
            if final is not None:
                await self._store_episode(query, final)
                yield FinalAnswerEvent(answer=final)
                return

            action_name, action_input = _parse_action(full_response)
            thought = _parse_thought(full_response)

            if thought:
                yield ThoughtEvent(thought=thought)

            if not action_name:
                final_answer = full_response.strip()
                await self._store_episode(query, final_answer)
                yield FinalAnswerEvent(answer=final_answer)
                return

            yield ActionEvent(tool=action_name, tool_input=action_input)

            try:
                tool = self._registry.get(action_name)
                result = await tool.run(input=action_input)
                observation = str(result)
            except Exception as e:
                observation = f"Error: {e}"
                yield ErrorEvent(error=str(e))

            yield ObservationEvent(observation=observation, tool=action_name)

            self._scratchpad.add_step(
                AgentStep(
                    thought=thought,
                    action=action_name,
                    action_input=action_input,
                    observation=observation,
                )
            )

        fallback = "I was unable to find the answer within the allowed number of steps."
        await self._store_episode(query, fallback)
        yield FinalAnswerEvent(answer=fallback)

    @property
    def memory(self) -> AgentScratchpad:
        return self._scratchpad

    @property
    def persistent_memory(self) -> PersistentAgentMemory | None:
        return self._persistent_memory
