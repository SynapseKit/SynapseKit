from __future__ import annotations

import re
from collections.abc import AsyncGenerator

from ..llm.base import BaseLLM
from .base import BaseTool
from .memory import AgentMemory, AgentStep
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
    """
    Reasoning + Acting agent.

    Loops: Thought → Action → Observation → repeat until Final Answer.
    Works with any BaseLLM — no native function-calling required.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool],
        max_iterations: int = 10,
        memory: AgentMemory | None = None,
    ) -> None:
        self._llm = llm
        self._registry = ToolRegistry(tools)
        self._max_iterations = max_iterations
        self._memory = memory or AgentMemory(max_steps=max_iterations)

    def _build_system_prompt(self) -> str:
        return _REACT_SYSTEM.format(tools=self._registry.describe())

    def _build_messages(self, query: str) -> list[dict]:
        scratchpad = self._memory.format_scratchpad()
        user_content = f"Question: {query}"
        if scratchpad:
            user_content += f"\n\n{scratchpad}"
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]

    async def run(self, query: str) -> str:
        """Run the ReAct loop and return the final answer."""
        self._memory.clear()

        for _ in range(self._max_iterations):
            messages = self._build_messages(query)
            response = await self._llm.generate_with_messages(messages)

            # Check for final answer first
            final = _parse_final_answer(response)
            if final is not None:
                return final

            # Parse action
            action_name, action_input = _parse_action(response)
            thought = _parse_thought(response)

            if not action_name:
                # LLM didn't follow format — treat whole response as final answer
                return response.strip()

            # Execute tool
            try:
                tool = self._registry.get(action_name)
                result = await tool.run(input=action_input)
                observation = str(result)
            except KeyError as e:
                observation = f"Error: {e}"
            except Exception as e:
                observation = f"Tool error: {e}"

            self._memory.add_step(
                AgentStep(
                    thought=thought,
                    action=action_name,
                    action_input=action_input,
                    observation=observation,
                )
            )

        return "I was unable to find the answer within the allowed number of steps."

    async def stream(self, query: str) -> AsyncGenerator[str]:
        """
        Stream the final answer. Intermediate tool calls run silently.
        Yields the final answer string (may be multi-token on last LLM call).
        """
        answer = await self.run(query)
        for word in answer.split(" "):
            yield word + " "

    @property
    def memory(self) -> AgentMemory:
        return self._memory
