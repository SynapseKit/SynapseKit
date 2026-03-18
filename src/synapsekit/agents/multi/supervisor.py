"""Supervisor agent -- coordinates multiple worker agents."""

from __future__ import annotations

from typing import Any

from ...llm.base import BaseLLM
from ..executor import AgentExecutor


class WorkerAgent:
    """A named worker agent with a role description."""

    def __init__(self, name: str, role: str, executor: AgentExecutor) -> None:
        self.name = name
        self.role = role
        self.executor = executor


class SupervisorAgent:
    """Supervisor that delegates tasks to worker agents.

    The supervisor LLM decides which worker to delegate to based on the task,
    collects results, and synthesizes a final response.

    Usage::

        research_exec = AgentExecutor(AgentConfig(llm=llm, tools=[WebSearchTool()]))
        math_exec = AgentExecutor(AgentConfig(llm=llm, tools=[CalculatorTool()]))

        supervisor = SupervisorAgent(
            llm=llm,
            workers=[
                WorkerAgent("researcher", "Search the web for information", research_exec),
                WorkerAgent("mathematician", "Solve math problems", math_exec),
            ],
        )
        result = await supervisor.run("What is the population of France times 42?")
    """

    def __init__(
        self,
        llm: BaseLLM,
        workers: list[WorkerAgent],
        max_rounds: int = 5,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._workers = {w.name: w for w in workers}
        self._max_rounds = max_rounds
        self._system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        worker_descriptions = "\n".join(f"- **{w.name}**: {w.role}" for w in self._workers.values())
        return (
            "You are a supervisor agent that delegates tasks to specialized workers.\n\n"
            "Available workers:\n"
            f"{worker_descriptions}\n\n"
            "To delegate, respond with EXACTLY this format:\n"
            "DELEGATE: worker_name | task description\n\n"
            "When you have enough information to answer the original question, "
            "respond with:\n"
            "FINAL: your complete answer\n\n"
            "Always delegate to the most appropriate worker. "
            "You may delegate multiple times before giving a final answer."
        )

    async def run(self, query: str) -> str:
        """Run the supervisor with the given query."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]

        for _ in range(self._max_rounds):
            response = await self._llm.generate("\n".join(m["content"] for m in messages))

            if response.strip().startswith("FINAL:"):
                return response.strip()[6:].strip()

            if response.strip().startswith("DELEGATE:"):
                delegation = response.strip()[9:].strip()
                parts = delegation.split("|", 1)
                if len(parts) == 2:
                    worker_name = parts[0].strip()
                    task = parts[1].strip()

                    worker = self._workers.get(worker_name)
                    if worker:
                        worker_result = await worker.executor.run(task)
                        messages.append({"role": "assistant", "content": response})
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Result from {worker_name}: {worker_result}",
                            }
                        )
                        continue
                    else:
                        messages.append({"role": "assistant", "content": response})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Worker '{worker_name}' not found. "
                                    f"Available: {', '.join(self._workers)}"
                                ),
                            }
                        )
                        continue

            # If response doesn't match expected format, treat as final
            return response

        return "Max rounds reached without a final answer."
