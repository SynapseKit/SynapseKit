"""Marketplace agent execution through the PC sandbox boundary."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from synapsekit.sandbox import PCSandbox


def test_installed_agent_runs_from_private_sandbox_runtime(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = tmp_path / "host"
        host.mkdir()
        agent = tmp_path / "installed-agent"
        agent.mkdir()
        (agent / "entry.py").write_text(
            "import sys\nprint(sys.argv[sys.argv.index('--prompt') + 1])\n",
            encoding="utf-8",
        )
        sandbox = PCSandbox(
            base=host,
            backend="fake",
            state_dir=tmp_path / "sessions",
        )
        try:
            result = await sandbox.run(
                agent,
                SimpleNamespace(name="example", entrypoint="entry.py"),
                "inspect safely",
                python_executable=sys.executable,
            )
            assert result.ok
            assert result.stdout.strip() == "inspect safely"

            diff = await sandbox.diff()
            assert diff.preview().changes == 0
        finally:
            await sandbox.discard()

    asyncio.run(scenario())
