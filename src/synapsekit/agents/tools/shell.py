from __future__ import annotations

import asyncio
import os
import shlex
from typing import Any

from ..base import BaseTool, ToolResult

# Characters that let a single command string spawn or chain other commands.
# If any appear while an allow-list is enforced, the command is rejected rather
# than executed, closing the ``echo hi & curl evil`` allow-list bypass.
_SHELL_METACHARACTERS = ("&", "|", ";", "`", "$(", "${", ">", "<", "\n", "\r", "(", ")")


def _portable_builtin(argv: list[str]) -> ToolResult | None:
    """Implement the few POSIX shell builtins used as direct commands on Windows.

    ``create_subprocess_exec`` deliberately avoids a shell, so Windows cannot
    resolve ``echo``, ``true``, or ``false``. Emulating those commands keeps
    direct execution and the allow-list boundary intact; no user-provided text
    is ever passed to ``cmd.exe``.
    """

    if os.name != "nt":
        return None
    command = argv[0].casefold()
    if command == "echo":
        values = argv[1:]
        newline = True
        if values and values[0] == "-n":
            values = values[1:]
            newline = False
        return ToolResult(output=" ".join(values) + ("\n" if newline else ""))
    if command == "true":
        return ToolResult(output="")
    if command == "false":
        return ToolResult(output="", error="Exit code 1:")
    return None


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

        try:
            argv = shlex.split(target)
        except ValueError as e:
            return ToolResult(output="", error=f"Invalid command: {e}")

        if not argv:
            return ToolResult(output="", error="No command provided.")

        if self.allowed_commands is not None:
            if argv[0] not in self.allowed_commands:
                return ToolResult(
                    output="",
                    error=f"Command {argv[0]!r} is not in the allowed list.",
                )
            # When an allow-list is enforced, reject shell metacharacters that
            # could chain a second (unlisted) command. The allow-list only vets
            # argv[0]; without this a string like "echo hi & curl evil" would
            # slip past on any code path that reaches a shell.
            metachar = next((m for m in _SHELL_METACHARACTERS if m in target), None)
            if metachar is not None:
                return ToolResult(
                    output="",
                    error=(
                        f"Command contains disallowed shell metacharacter "
                        f"{metachar!r} while an allow-list is enforced."
                    ),
                )

        builtin_result = _portable_builtin(argv)
        if builtin_result is not None:
            return builtin_result

        try:
            # Always exec the parsed argv (never shell=True). This runs the
            # command directly without a shell interpreter, so metacharacters in
            # the raw string cannot spawn additional commands on any platform.
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            output = stdout.decode() if stdout else ""
            err = stderr.decode() if stderr else ""

            # Ensure subprocess is fully cleaned up
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

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
