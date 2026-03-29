from __future__ import annotations

import contextlib
import io
import logging
import math
import multiprocessing
import platform
import signal
import sys
from typing import Any

from ..base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class _CodeTimeoutError(Exception):
    """Raised when code execution times out."""


def _exec_with_capture(code: str, namespace: dict, output_queue: multiprocessing.Queue) -> None:
    """Execute code in a subprocess and send results back via queue.

    Used for Windows timeout implementation with multiprocessing.
    """
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()

    try:
        exec(code, namespace)
        output = buf.getvalue()
        # Send back (success, output, error, namespace)
        output_queue.put((True, output, None, namespace))
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        output_queue.put((False, "", error_msg, namespace))
    finally:
        sys.stdout = old_stdout


class PythonREPLTool(BaseTool):
    """
    Execute arbitrary Python code and capture stdout.

    Timeout behavior:
    - Unix/Linux: Uses signal.SIGALRM (fast, full namespace persistence)
    - Windows: Uses multiprocessing.Process (reliable timeout, namespace limited to picklable objects)

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

        is_windows = platform.system() == "Windows"

        if is_windows:
            return await self._run_windows(src)
        else:
            return await self._run_unix(src)

    async def _run_unix(self, code: str) -> ToolResult:
        """Execute code with signal.SIGALRM timeout (Unix/Linux)."""
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()

        def timeout_handler(signum, frame):
            raise _CodeTimeoutError(f"Code execution timed out after {self.timeout} seconds")

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max(1, math.ceil(self.timeout)))
            try:
                exec(code, self._namespace)
            finally:
                signal.alarm(0)  # Cancel alarm

            output = buf.getvalue()
            return ToolResult(output=output or "(no output)")
        except _CodeTimeoutError as e:
            return ToolResult(output="", error=str(e))
        except Exception as e:
            return ToolResult(output="", error=f"{type(e).__name__}: {e}")
        finally:
            sys.stdout = old_stdout
            signal.alarm(0)  # Ensure alarm is cancelled

    async def _run_windows(self, code: str) -> ToolResult:
        """Execute code with multiprocessing timeout (Windows).

        Note: Namespace persistence limited to picklable objects.
        """
        output_queue: multiprocessing.Queue = multiprocessing.Queue()

        # Create process to execute code
        process = multiprocessing.Process(
            target=_exec_with_capture,
            args=(code, self._namespace.copy(), output_queue)
        )

        process.start()
        process.join(timeout=self.timeout)

        if process.is_alive():
            # Timeout occurred
            process.terminate()
            process.join(timeout=1.0)  # Wait for graceful termination
            if process.is_alive():
                process.kill()  # Force kill if still running
            return ToolResult(output="", error=f"Code execution timed out after {self.timeout} seconds")

        # Process completed, get results
        if not output_queue.empty():
            success, output, error, updated_namespace = output_queue.get()

            # Update namespace with picklable objects from subprocess
            # This allows persistence of basic types but not complex objects
            with contextlib.suppress(Exception):
                self._namespace.update(updated_namespace)

            if success:
                return ToolResult(output=output or "(no output)")
            else:
                return ToolResult(output="", error=error)
        else:
            # Process died without sending results
            return ToolResult(output="", error="Code execution failed unexpectedly")

    def reset(self) -> None:
        """Clear the persistent namespace between runs."""
        self._namespace.clear()
