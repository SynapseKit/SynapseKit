"""Step event types for streaming agent reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ThoughtEvent:
    """Agent's internal reasoning step."""

    thought: str
    type: str = "thought"


@dataclass
class ActionEvent:
    """Agent decided to call a tool."""

    tool: str
    tool_input: str | dict[str, Any] = ""
    type: str = "action"


@dataclass
class ObservationEvent:
    """Result from a tool execution."""

    observation: str
    tool: str = ""
    type: str = "observation"


@dataclass
class TokenEvent:
    """A single streamed token from the LLM."""

    token: str
    type: str = "token"


@dataclass
class FinalAnswerEvent:
    """The agent's final answer."""

    answer: str
    type: str = "final_answer"


@dataclass
class ErrorEvent:
    """An error occurred during agent execution."""

    error: str
    type: str = "error"


StepEvent = (
    ThoughtEvent | ActionEvent | ObservationEvent | TokenEvent | FinalAnswerEvent | ErrorEvent
)
