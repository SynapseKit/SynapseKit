from __future__ import annotations

import io
import sys
from typing import Any

from ..base import BaseTool, ToolResult


class PythonREPLTool(BaseTool):
    """
    Execute arbitrary Python code and capture stdout.

    Warning: This executes real Python code. Only use in trusted environments.
    """

    name = "python_repl"
    description = (
        "Execute Python code and return its output. "
        "Input: a Python code string. Use print() to produce output. "
        "WARNING: executes real Python — only use in trusted environments."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            }
        },
        "required": ["code"],
    }

    def __init__(self) -> None:
        self._namespace: dict = {}

    async def run(self, code: str = "", **kwargs: Any) -> ToolResult:
        src = code or kwargs.get("input", "")
        if not src:
            return ToolResult(output="", error="No code provided.")

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            exec(src, self._namespace)  # noqa: S102
            output = buf.getvalue()
            return ToolResult(output=output or "(no output)")
        except Exception as e:
            return ToolResult(output="", error=f"{type(e).__name__}: {e}")
        finally:
            sys.stdout = old_stdout

    def reset(self) -> None:
        """Clear the persistent namespace between runs."""
        self._namespace.clear()
