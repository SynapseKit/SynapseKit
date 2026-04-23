"""@tool decorator — turn any function into a BaseTool."""

from __future__ import annotations

import inspect
from typing import Any

from .base import BaseTool, ToolResult


def tool(
    name: str | callable | None = None,
    description: str | None = None,
) -> Any:
    """Decorator that wraps a plain function into a BaseTool.

    Usage::

        @tool(name="add", description="Add two numbers")
        def add(a: int, b: int) -> str:
            return str(a + b)

        # Or minimal — infers name and description from function:
        @tool
        def multiply(a: int, b: int) -> str:
            \"\"\"Multiply two numbers.\"\"\"
            return str(a * b)

    The decorated function can be sync or async.
    """
    if callable(name):
        return tool()(name)

    def decorator(fn: Any) -> BaseTool:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or f"{tool_name} tool"
        params = _build_params(fn)
        is_async = inspect.iscoroutinefunction(fn)

        class _DynamicTool(BaseTool):
            async def run(self, **kwargs: Any) -> ToolResult:
                try:
                    result = fn(**kwargs) if not is_async else await fn(**kwargs)
                    return ToolResult(output=str(result))
                except Exception as e:
                    return ToolResult(output="", error=str(e))

        _DynamicTool.name = tool_name
        _DynamicTool.description = tool_desc
        _DynamicTool.parameters = params
        _DynamicTool.__name__ = tool_name
        _DynamicTool.__qualname__ = tool_name

        return _DynamicTool()

    return decorator


_TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


def _build_params(fn: Any) -> dict:
    """Build JSON Schema parameters from function signature."""
    sig = inspect.signature(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        prop: dict[str, str] = {}
        if param.annotation != inspect.Parameter.empty:
            prop["type"] = _TYPE_MAP.get(param.annotation, "string")
        else:
            prop["type"] = "string"
        properties[param_name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
