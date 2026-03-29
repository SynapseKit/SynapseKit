from __future__ import annotations

import io
import logging
import platform
import signal
import sys
from typing import Any

from ..base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when code execution times out."""


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

    def __init__(self, timeout: float = 5.0) -> None:
        logger.warning(
            "PythonREPLTool executes arbitrary Python code. "
            "Only use in trusted environments with controlled input. "
            "Malicious code can access files, network, and system resources."
        )
        self._namespace: dict = {}
        self.timeout = timeout

    async def run(self, code: str = "", **kwargs: Any) -> ToolResult:
        src = code or kwargs.get("input", "")
        if not src:
            return ToolResult(output="", error="No code provided.")

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()

        # Use signal-based timeout on Unix
        # Note: Windows doesn't support signal.SIGALRM, so timeout is best-effort there
        is_unix = platform.system() != "Windows"

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution timed out after {self.timeout} seconds")

        try:
            if is_unix:
                # Unix: use signal.alarm for reliable timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))
                try:
                    exec(src, self._namespace)
                finally:
                    signal.alarm(0)  # Cancel alarm
            else:
                # Windows: no reliable way to interrupt exec(), run without timeout
                # The timeout parameter is documented but not enforced on Windows
                exec(src, self._namespace)

            output = buf.getvalue()
            return ToolResult(output=output or "(no output)")
        except TimeoutError as e:
            return ToolResult(output="", error=str(e))
        except Exception as e:
            return ToolResult(output="", error=f"{type(e).__name__}: {e}")
        finally:
            sys.stdout = old_stdout
            if is_unix:
                signal.alarm(0)  # Ensure alarm is cancelled

    def reset(self) -> None:
        """Clear the persistent namespace between runs."""
        self._namespace.clear()
