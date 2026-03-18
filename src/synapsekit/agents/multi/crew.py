"""Crew -- role-based multi-agent teams."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from ...llm.base import BaseLLM
from ..base import BaseTool
from ..executor import AgentConfig, AgentExecutor


@dataclass
class CrewAgent:
    """An agent with a role in a crew."""

    name: str
    role: str
    goal: str
    llm: BaseLLM
    tools: list[BaseTool] = field(default_factory=list)
    backstory: str = ""


@dataclass
class Task:
    """A task assigned to a crew agent."""

    description: str
    agent: str
    expected_output: str = ""
    context_from: list[str] = field(default_factory=list)


@dataclass
class CrewResult:
    """Result of crew execution."""

    output: str
    task_results: dict[str, str] = field(default_factory=dict)


class Crew:
    """Role-based multi-agent team execution.

    Usage::

        crew = Crew(
            agents=[
                CrewAgent(
                    "researcher", "Research Analyst", "Find accurate information",
                    llm, [WebSearchTool()],
                ),
                CrewAgent("writer", "Content Writer", "Write clear content", llm),
            ],
            tasks=[
                Task(
                    "Research the latest AI trends",
                    agent="researcher",
                    expected_output="A summary of trends",
                ),
                Task(
                    "Write a blog post about the trends",
                    agent="writer",
                    context_from=["researcher"],
                ),
            ],
            process="sequential",
        )
        result = await crew.run()
    """

    def __init__(
        self,
        agents: list[CrewAgent],
        tasks: list[Task],
        process: Literal["sequential", "parallel"] = "sequential",
        verbose: bool = False,
    ) -> None:
        self._agents = {a.name: a for a in agents}
        self._tasks = tasks
        self._process = process
        self._verbose = verbose

    def _build_executor(self, agent: CrewAgent) -> AgentExecutor:
        system_prompt = f"You are a {agent.role}.\nYour goal: {agent.goal}\n"
        if agent.backstory:
            system_prompt += f"Background: {agent.backstory}\n"

        config = AgentConfig(
            llm=agent.llm,
            tools=agent.tools,
            system_prompt=system_prompt,
        )
        return AgentExecutor(config)

    async def run(self) -> CrewResult:
        task_results: dict[str, str] = {}

        if self._process == "sequential":
            for task in self._tasks:
                agent = self._agents.get(task.agent)
                if not agent:
                    raise ValueError(f"Unknown agent: {task.agent!r}")

                # Build context from previous task results
                context = ""
                for ctx_agent in task.context_from:
                    if ctx_agent in task_results:
                        context += f"\nContext from {ctx_agent}:\n{task_results[ctx_agent]}\n"

                prompt = task.description
                if context:
                    prompt = f"{context}\n\nTask: {task.description}"
                if task.expected_output:
                    prompt += f"\n\nExpected output format: {task.expected_output}"

                executor = self._build_executor(agent)
                result = await executor.run(prompt)
                task_results[task.agent] = result

        elif self._process == "parallel":
            # Group tasks by dependency -- tasks with no context_from run first
            no_deps = [t for t in self._tasks if not t.context_from]
            with_deps = [t for t in self._tasks if t.context_from]

            # Run independent tasks in parallel
            async def _run_task(task: Task) -> tuple[str, str]:
                agent = self._agents.get(task.agent)
                if not agent:
                    raise ValueError(f"Unknown agent: {task.agent!r}")
                executor = self._build_executor(agent)
                prompt = task.description
                if task.expected_output:
                    prompt += f"\n\nExpected output format: {task.expected_output}"
                result = await executor.run(prompt)
                return task.agent, result

            results = await asyncio.gather(*[_run_task(t) for t in no_deps])
            for agent_name, result in results:
                task_results[agent_name] = result

            # Run dependent tasks sequentially
            for task in with_deps:
                agent = self._agents.get(task.agent)
                if not agent:
                    raise ValueError(f"Unknown agent: {task.agent!r}")

                context = ""
                for ctx_agent in task.context_from:
                    if ctx_agent in task_results:
                        context += f"\nContext from {ctx_agent}:\n{task_results[ctx_agent]}\n"

                prompt = task.description
                if context:
                    prompt = f"{context}\n\nTask: {task.description}"
                if task.expected_output:
                    prompt += f"\n\nExpected output format: {task.expected_output}"

                executor = self._build_executor(agent)
                result = await executor.run(prompt)
                task_results[task.agent] = result

        # Last task result is the final output
        final_output = task_results.get(self._tasks[-1].agent, "") if self._tasks else ""

        return CrewResult(output=final_output, task_results=task_results)
