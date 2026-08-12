from __future__ import annotations

import asyncio
from pathlib import Path

from synapsekit.audit import Verdict, verify
from synapsekit.shell import ShellHistory, ShellSession, generate_signing_key, load_signing_policy


def test_safe_command_runs_without_a_signing_key(tmp_path: Path) -> None:
    session = ShellSession(
        cwd=tmp_path,
        shell="bash",
        mesh=None,
        history=ShellHistory(tmp_path / "history.sqlite3"),
    )

    result = asyncio.run(session.run("echo hello"))

    assert result.ok
    assert result.commands[0].stdout.strip() == "hello"


def test_destructive_command_fails_closed_when_unsigned(tmp_path: Path) -> None:
    session = ShellSession(
        cwd=tmp_path,
        shell="bash",
        mesh=None,
        history=ShellHistory(tmp_path / "history.sqlite3"),
    )

    result = asyncio.run(session.run("git reset --hard HEAD", assume_yes=True))

    assert result.aborted
    assert result.error is not None
    assert "--signing-key" in result.error


def test_denied_destructive_command_writes_signed_receipt(tmp_path: Path) -> None:
    private = tmp_path / "shell.key"
    generate_signing_key(private)
    policy = load_signing_policy(private)
    session = ShellSession(
        cwd=tmp_path,
        shell="bash",
        mesh=None,
        history=ShellHistory(tmp_path / "history.sqlite3"),
        signing_policy=policy,
        audit_dir=tmp_path / "audit",
    )

    async def deny(_step: object, _preview: str) -> bool:
        return False

    result = asyncio.run(session.run("git reset --hard HEAD", confirm=deny))

    assert result.aborted
    assert result.audit_path is not None
    verification = verify(
        result.audit_path,
        trusted_keys={policy.provider.key_id: policy.provider.public_key_bytes()},
    )
    assert verification.verdict is Verdict.MATCH


def test_history_redacts_secret_shaped_values(tmp_path: Path) -> None:
    history = ShellHistory(tmp_path / "history.sqlite3")

    async def exercise() -> list[dict[str, object]]:
        await history.record(
            cwd=str(tmp_path),
            input_text="echo api_key=top-secret",
            commands=["echo api_key=top-secret"],
            ok=True,
        )
        return await history.search("api_key")

    rows = asyncio.run(exercise())

    assert rows
    assert "top-secret" not in str(rows[0])
    assert "<redacted>" in str(rows[0])
