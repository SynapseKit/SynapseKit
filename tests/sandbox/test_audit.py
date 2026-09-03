"""Audit integration for sandbox lifecycle traces."""

from __future__ import annotations

import asyncio
from pathlib import Path

from synapsekit.audit import SigningPolicy
from synapsekit.sandbox import PCSandbox


def test_sandbox_exports_a_signed_audit_bundle(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = tmp_path / "host"
        host.mkdir()
        sandbox = PCSandbox(
            base=host,
            backend="fake",
            state_dir=tmp_path / "sessions",
        )
        environment = await sandbox.start()
        try:
            output = environment.export_audit_bundle(
                tmp_path / "sandbox.audit.zip",
                SigningPolicy.ed25519(),
            )
            assert output.is_file()
            assert output.stat().st_size > 0
        finally:
            await sandbox.discard()

    asyncio.run(scenario())
