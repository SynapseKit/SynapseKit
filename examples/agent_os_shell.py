"""Run a safe local Agent OS Shell interaction."""

from __future__ import annotations

import asyncio
from pathlib import Path

from synapsekit.shell import ShellHistory, ShellSession


async def main() -> None:
    root = Path.cwd()
    session = ShellSession(
        cwd=root,
        mesh=None,
        history=ShellHistory(root / ".synapsekit_shell_history.sqlite3"),
    )
    result = await session.run("git status --short --branch")
    print(result.plan.summary)
    for command in result.commands:
        print(command.stdout or command.stderr)


if __name__ == "__main__":
    asyncio.run(main())
