"""Agent handoff -- transfer control between agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..executor import AgentExecutor


@dataclass
class HandoffResult:
    """Result of a handoff chain execution."""

    final_output: str
    history: list[dict[str, Any]] = field(default_factory=list)


class Handoff:
    """Defines a handoff from one agent to another.

    Usage::

        h = Handoff(
            target="specialist",
            condition=lambda result: "need expert" in result.lower(),
            transform=lambda result: f"The previous agent said: {result}. Please help.",
        )
    """

    def __init__(
        self,
        target: str,
        condition: Callable[[str], bool] | None = None,
        transform: Callable[[str], str] | None = None,
    ) -> None:
        self.target = target
        self.condition: Callable[[str], bool] = condition or (lambda _: True)
        self.transform: Callable[[str], str] = transform or (lambda x: x)


class HandoffChain:
    """Execute agents with handoff support.

    Usage::

        chain = HandoffChain()
        chain.add_agent("triage", triage_executor, handoffs=[
            Handoff("billing", condition=lambda r: "billing" in r.lower()),
            Handoff("technical", condition=lambda r: "technical" in r.lower()),
        ])
        chain.add_agent("billing", billing_executor)
        chain.add_agent("technical", technical_executor)

        result = await chain.run("triage", "I have a billing question about my invoice")
    """

    def __init__(self, max_handoffs: int = 10) -> None:
        self._agents: dict[str, tuple[AgentExecutor, list[Handoff]]] = {}
        self._max_handoffs = max_handoffs

    def add_agent(
        self,
        name: str,
        executor: AgentExecutor,
        handoffs: list[Handoff] | None = None,
    ) -> None:
        self._agents[name] = (executor, handoffs or [])

    async def run(self, start_agent: str, query: str) -> HandoffResult:
        current_agent = start_agent
        current_input = query
        history: list[dict[str, Any]] = []

        for _ in range(self._max_handoffs):
            if current_agent not in self._agents:
                raise ValueError(
                    f"Unknown agent: {current_agent!r}. Available: {', '.join(self._agents)}"
                )

            executor, handoffs = self._agents[current_agent]
            result = await executor.run(current_input)

            history.append(
                {
                    "agent": current_agent,
                    "input": current_input,
                    "output": result,
                }
            )

            # Check handoff conditions
            handed_off = False
            for handoff in handoffs:
                if handoff.condition(result):
                    current_agent = handoff.target
                    current_input = handoff.transform(result)
                    handed_off = True
                    break

            if not handed_off:
                return HandoffResult(final_output=result, history=history)

        return HandoffResult(
            final_output=history[-1]["output"] if history else "",
            history=history,
        )
