from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ..agents.base import BaseTool, ToolResult
from ..agents.tool_decorator import tool
from .types import ConstraintSet, SymbolicBackend


class VerifiedTool(BaseTool):
    """Tool wrapper that verifies a tool result with a symbolic backend."""

    def __init__(
        self,
        inner: BaseTool,
        verifier: SymbolicBackend,
        constraint_builder: Callable[[dict[str, Any], str], ConstraintSet],
        *,
        timeout_seconds: float | None = 10.0,
    ) -> None:
        self._inner = inner
        self._verifier = verifier
        self._constraint_builder = constraint_builder
        self._timeout_seconds = timeout_seconds
        self.name = inner.name
        self.description = inner.description
        self.parameters = inner.parameters

    async def run(self, **kwargs: Any) -> ToolResult:
        result = await self._inner.run(**kwargs)
        if result.error:
            return result
        constraints = self._constraint_builder(kwargs, result.output)
        proof = await self._verifier.verify(
            constraints,
            result.output,
            await self._verifier.solve(constraints, timeout_seconds=self._timeout_seconds),
            timeout_seconds=self._timeout_seconds,
        )
        if proof.is_verified:
            return ToolResult(output=result.output, metadata={"proof": proof})
        return ToolResult(
            output="",
            error=proof.error or "Tool result failed symbolic verification.",
            metadata={"proof": proof},
        )


def verified_tool(
    verifier: SymbolicBackend,
    constraint_builder: Callable[[dict[str, Any], str], ConstraintSet],
    *,
    name: str | None = None,
    description: str | None = None,
    timeout_seconds: float | None = 10.0,
) -> Callable[[Any], VerifiedTool]:
    """Decorate a function or BaseTool so its output is solver-verified."""

    def decorator(fn_or_tool: Any) -> VerifiedTool:
        if isinstance(fn_or_tool, BaseTool):
            inner = fn_or_tool
        elif inspect.isfunction(fn_or_tool) or inspect.iscoroutinefunction(fn_or_tool):
            inner = tool(name=name, description=description)(fn_or_tool)
        else:
            raise TypeError("@verified_tool expects a function or BaseTool instance.")

        return VerifiedTool(
            inner,
            verifier,
            constraint_builder,
            timeout_seconds=timeout_seconds,
        )

    return decorator
