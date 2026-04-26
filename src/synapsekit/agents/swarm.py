"""SwarmAgent: dynamic sub-agent spawning based on task complexity."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..llm.base import BaseLLM
from .base import BaseTool
from .function_calling import FunctionCallingAgent

_COMPLEXITY_PROMPT = """\
Rate the complexity of the following task on a scale from 0.0 (trivial) to 1.0 (highly complex).
Respond with ONLY a JSON object with a single key "score" and a float value, e.g. {{"score": 0.8}}.

Task: {task}
"""

_DECOMPOSE_PROMPT = """\
Decompose the following task into at most {max_subtasks} smaller, independent subtasks.
Respond with ONLY a JSON array of strings, each string being a subtask description.

Task: {task}
"""

_SYNTHESISE_PROMPT = """\
You are synthesizing the results of multiple subtasks into a single, coherent final answer.

Original task: {task}

Subtask results:
{results}

Provide a concise, complete final answer.
"""


class SwarmAgent:
    """Dynamically spawns sub-agents when task complexity exceeds a threshold.

    Steps:
    1. Score task complexity (0.0-1.0) with the LLM.
    2. If score < ``spawn_threshold``, handle directly with a
       ``FunctionCallingAgent``.
    3. Otherwise, decompose into subtasks, spawn up to ``max_agents``
       ``FunctionCallingAgent`` instances in parallel, then synthesise
       a final answer.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        max_agents: int = 5,
        spawn_threshold: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        if max_agents < 1:
            raise ValueError("max_agents must be >= 1")
        if not (0.0 <= spawn_threshold <= 1.0):
            raise ValueError("spawn_threshold must be between 0.0 and 1.0")
        self._llm = llm
        self._tools: list[BaseTool] = tools or []
        self._max_agents = max_agents
        self._spawn_threshold = spawn_threshold
        self._system_prompt = system_prompt

    def __repr__(self) -> str:
        return (
            f"SwarmAgent(llm={type(self._llm).__name__!r}, "
            f"tools={len(self._tools)}, "
            f"max_agents={self._max_agents}, "
            f"spawn_threshold={self._spawn_threshold})"
        )

    async def _score_complexity(self, task: str) -> float:
        prompt = _COMPLEXITY_PROMPT.format(task=task)
        raw = await self._llm.generate(prompt)
        try:
            data = json.loads(raw.strip())
            score = float(data.get("score", 0.5))
        except (json.JSONDecodeError, ValueError, TypeError):
            # Attempt to extract a bare float from the response
            import re

            m = re.search(r"\b([01](\.\d+)?|\.\d+)\b", raw)
            score = float(m.group(1)) if m else 0.5
        return max(0.0, min(1.0, score))

    async def _decompose(self, task: str) -> list[str]:
        prompt = _DECOMPOSE_PROMPT.format(task=task, max_subtasks=self._max_agents)
        raw = await self._llm.generate(prompt)
        try:
            subtasks: Any = json.loads(raw.strip())
            if isinstance(subtasks, list):
                return [str(s) for s in subtasks][: self._max_agents]
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: treat whole response as a single subtask
        return [task]

    def _make_agent(self) -> FunctionCallingAgent:
        return FunctionCallingAgent(
            llm=self._llm,
            tools=self._tools,
            system_prompt=self._system_prompt,
        )

    async def _run_subtask(self, subtask: str) -> str:
        agent = self._make_agent()
        try:
            return await agent.run(subtask)
        except Exception as exc:
            return f"[subtask error] {exc}"

    async def _synthesise(self, task: str, results: list[str]) -> str:
        formatted = "\n".join(f"- {r}" for r in results)
        prompt = _SYNTHESISE_PROMPT.format(task=task, results=formatted)
        return await self._llm.generate(prompt)

    async def run(self, task: str) -> str:
        """Execute the swarm agent on *task* and return the final answer."""
        score = await self._score_complexity(task)

        if score < self._spawn_threshold:
            agent = self._make_agent()
            try:
                return await agent.run(task)
            except Exception:
                # FunctionCallingAgent requires tool-call support; fall back
                return await self._llm.generate(task)

        subtasks = await self._decompose(task)
        if not subtasks:
            subtasks = [task]

        coroutines = [self._run_subtask(st) for st in subtasks]
        results: list[str] = list(await asyncio.gather(*coroutines))
        return await self._synthesise(task, results)
