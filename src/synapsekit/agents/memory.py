from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentStep:
    """One complete Thought → Action → Observation cycle."""

    thought: str
    action: str
    action_input: str
    observation: str


class AgentMemory:
    """Scratchpad that records agent steps for the current run."""

    def __init__(self, max_steps: int = 20) -> None:
        self._max_steps = max_steps
        self._steps: list[AgentStep] = []

    def add_step(self, step: AgentStep) -> None:
        self._steps.append(step)

    @property
    def steps(self) -> list[AgentStep]:
        return list(self._steps)

    def format_scratchpad(self) -> str:
        """Format all steps as a ReAct scratchpad string."""
        parts = []
        for step in self._steps:
            parts.append(f"Thought: {step.thought}")
            parts.append(f"Action: {step.action}")
            parts.append(f"Action Input: {step.action_input}")
            parts.append(f"Observation: {step.observation}")
        return "\n".join(parts)

    def is_full(self) -> bool:
        return len(self._steps) >= self._max_steps

    def clear(self) -> None:
        self._steps.clear()

    def __len__(self) -> int:
        return len(self._steps)
