from __future__ import annotations

from collections.abc import Iterator

from .base import BaseTool


class ToolRegistry:
    """Lookup table mapping tool name → BaseTool instance."""

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            available = list(self._tools)
            raise KeyError(f"Tool {name!r} not found. Available: {available}")
        return self._tools[name]

    def schemas(self) -> list[dict]:
        """Return OpenAI-compatible function schemas for all tools."""
        return [t.schema() for t in self._tools.values()]

    def anthropic_schemas(self) -> list[dict]:
        """Return Anthropic-compatible tool schemas for all tools."""
        return [t.anthropic_schema() for t in self._tools.values()]

    def describe(self) -> str:
        """Human-readable description of all tools (for ReAct prompts)."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
