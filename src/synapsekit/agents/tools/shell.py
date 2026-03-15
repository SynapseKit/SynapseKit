from __future__ import annotations

import asyncio
from typing import Any

from ..base import BaseTool, ToolResult


class ShellTool(BaseTool):
    """Execute shell commands and return their output."""

    name = "shell"
    description = (
        "Execute a shell command and return stdout/stderr. "
        "Input: a command string. "
        "Optional: allowed_commands to restrict which commands can be run."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
        },
        "required": ["command"],
    }

    def __init__(
        self,
        timeout: int = 30,
        allowed_commands: list[str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.allowed_commands = allowed_commands

    async def run(self, command: str = "", **kwargs: Any) -> ToolResult:
        """Execute the shell command."""
        target = command or kwargs.get("input", "")
        if not target:
            return ToolResult(output="", error="No command provided.")

        if self.allowed_commands is not None:
            base_cmd = target.split()[0]
            if base_cmd not in self.allowed_commands:
                return ToolResult(
                    output="",
                    error=f"Command {base_cmd!r} is not in the allowed list.",
                )

        try:
            proc = await asyncio.create_subprocess_shell(
                target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            output = stdout.decode() if stdout else ""
            err = stderr.decode() if stderr else ""

            if proc.returncode != 0:
                return ToolResult(
                    output=output,
                    error=f"Exit code {proc.returncode}: {err}".strip(),
                )
            return ToolResult(output=output + err)
        except TimeoutError:
            return ToolResult(output="", error=f"Command timed out after {self.timeout}s.")
        except Exception as e:
            return ToolResult(output="", error=f"Shell error: {e}")
