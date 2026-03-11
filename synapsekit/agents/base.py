from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result returned by any tool execution."""

    output: str
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def __str__(self) -> str:
        return self.error if self.is_error else self.output


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str

    # JSON Schema for the tool's input parameters.
    # Subclasses must define this as a class attribute.
    parameters: dict = field(default_factory=dict)

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. kwargs come from the parsed Action Input."""
        ...

    def schema(self) -> dict:
        """OpenAI-compatible function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": getattr(self, "parameters", {"type": "object", "properties": {}}),
            },
        }

    def anthropic_schema(self) -> dict:
        """Anthropic-compatible tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": getattr(self, "parameters", {"type": "object", "properties": {}}),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
