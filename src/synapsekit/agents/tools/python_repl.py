from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import math
import multiprocessing
import pickle
import platform
import queue
import signal
import sys
from typing import Any

from ..base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class _CodeTimeoutError(Exception):
    """Raised when code execution times out."""


def _picklable_namespace(namespace: dict[str, Any]) -> dict[str, Any]:
    """Keep only serializable user bindings for the Windows child-process result."""

    result: dict[str, Any] = {}
    for key, value in namespace.items():
        if key == "__builtins__":
            continue
        try:
            pickle.dumps(value)
        except (pickle.PicklingError, TypeError, AttributeError):
            continue
        result[key] = value
    return result


def _exec_with_capture(code: str, namespace: dict, output_queue: multiprocessing.Queue) -> None:
    """Execute code in a subprocess and send results back via queue.

    Used for Windows timeout implementation with multiprocessing.
    """

    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()

    try:
        exec(code, namespace)
        output = buf.getvalue()
        # ``exec`` injects a large ``__builtins__`` mapping. Returning it
        # through a multiprocessing queue can deadlock the child while the
        # parent waits in ``join``; return only picklable user bindings.
        output_queue.put((True, output, None, _picklable_namespace(namespace)))
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        output_queue.put((False, "", error_msg, _picklable_namespace(namespace)))
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
        "WARNING: executes real Python code; only use in trusted environments."
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
            signal.signal(signal.SIGALRM, timeout_handler)  # type: ignore[attr-defined]
            signal.alarm(max(1, math.ceil(self.timeout)))  # type: ignore[attr-defined]
            try:
                exec(code, self._namespace)
            finally:
                signal.alarm(0)  # type: ignore[attr-defined]

            output = buf.getvalue()
            return ToolResult(output=output or "(no output)")
        except _CodeTimeoutError as e:
            return ToolResult(output="", error=str(e))
        except Exception as e:
            return ToolResult(output="", error=f"{type(e).__name__}: {e}")
        finally:
            sys.stdout = old_stdout
            signal.alarm(0)  # type: ignore[attr-defined]

    async def _run_windows(self, code: str) -> ToolResult:
        """Execute code with a child process and an enforced Windows timeout.

        The parent drains the result queue before joining the child. Joining
        first can deadlock when a Windows queue feeder is still flushing its
        payload, which made even a simple print appear to time out.
        """

        output_queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_exec_with_capture, args=(code, self._namespace.copy(), output_queue)
        )
        process.start()
        # Windows uses spawn, so importing the interpreter and test environment
        # can take longer than a short user-code timeout. Count that bounded
        # startup cost separately from the requested execution budget.
        wait_timeout = self.timeout + 5.0
        try:
            success, output, error, updated_namespace = await asyncio.to_thread(
                output_queue.get,
                True,
                wait_timeout,
            )
        except queue.Empty:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
            return ToolResult(
                output="", error=f"Code execution timed out after {self.timeout} seconds"
            )
        finally:
            if process.is_alive():
                process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            output_queue.close()
            output_queue.join_thread()

        with contextlib.suppress(Exception):
            self._namespace.update(updated_namespace)

        if success:
            return ToolResult(output=output or "(no output)")
        return ToolResult(output="", error=error)

    def reset(self) -> None:
        """Clear the persistent namespace between runs."""

        self._namespace.clear()
