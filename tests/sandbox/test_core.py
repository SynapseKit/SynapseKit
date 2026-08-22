"""Core PC Twin tests that do not require Docker or a VM runtime."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from synapsekit.sandbox import (
    ApplyConflictError,
    CallableEvalGate,
    DiffBundle,
    PCSandbox,
    SandboxSecurityError,
    SandboxState,
)
from synapsekit.sandbox.overlay import capture_manifest


def _run(coro):
    return asyncio.run(coro)


async def _open_fake(tmp_path: Path) -> tuple[PCSandbox, object]:
    base = tmp_path / "host"
    base.mkdir()
    (base / "project").mkdir()
    (base / "project" / "input.txt").write_text("before\n", encoding="utf-8")
    (base / ".env").write_text("SECRET=do-not-copy\n", encoding="utf-8")
    sandbox = PCSandbox(
        base=base,
        backend="fake",
        state_dir=tmp_path / "sessions",
        exclude=(".env",),
    )
    environment = await sandbox.start()
    return sandbox, environment


def test_snapshot_diff_eval_and_apply(tmp_path: Path) -> None:
    async def scenario() -> None:
        sandbox, environment = await _open_fake(tmp_path)
        try:
            assert not (environment.work_root / ".env").exists()
            (environment.work_root / "project" / "input.txt").write_text(
                "after\n", encoding="utf-8"
            )
            (environment.work_root / "project" / "new.txt").write_text("new\n", encoding="utf-8")

            diff = await environment.diff_against_host()
            assert diff.preview().modifications == 1
            assert diff.preview().additions == 1

            gate = CallableEvalGate(lambda current_diff, env: {"passed": True, "score": 1.0})
            receipt = await environment.evaluate(gate, diff)
            await environment.apply(diff, receipt)

            assert environment.state == SandboxState.APPLIED
            assert (environment.host_root / "project" / "input.txt").read_text() == "after\n"
            assert (environment.host_root / "project" / "new.txt").read_text() == "new\n"
        finally:
            await sandbox.discard()

    _run(scenario())


def test_file_deletion_is_diffed_and_applied(tmp_path: Path) -> None:
    async def scenario() -> None:
        sandbox, environment = await _open_fake(tmp_path)
        try:
            target = environment.work_root / "project" / "input.txt"
            target.unlink()
            diff = await environment.diff_against_host()
            assert diff.preview().deletions == 1
            receipt = await environment.evaluate(CallableEvalGate(lambda *_: True), diff)
            await environment.apply(diff, receipt)
            assert not (environment.host_root / "project" / "input.txt").exists()
        finally:
            await sandbox.discard()

    _run(scenario())


def test_apply_requires_matching_passing_receipt(tmp_path: Path) -> None:
    async def scenario() -> None:
        sandbox, environment = await _open_fake(tmp_path)
        try:
            (environment.work_root / "project" / "input.txt").write_text("after\n")
            diff = await environment.diff_against_host()
            rejected = await environment.evaluate(
                CallableEvalGate(lambda current_diff, env: False), diff
            )
            with pytest.raises(ApplyConflictError):
                await diff.apply(rejected)
        finally:
            await sandbox.discard()

    _run(scenario())


def test_host_conflict_is_detected_before_mutation(tmp_path: Path) -> None:
    async def scenario() -> None:
        sandbox, environment = await _open_fake(tmp_path)
        try:
            (environment.work_root / "project" / "input.txt").write_text("sandbox\n")
            diff = await environment.diff_against_host()
            receipt = await environment.evaluate(
                CallableEvalGate(lambda current_diff, env: True), diff
            )
            (environment.host_root / "project" / "input.txt").write_text("host changed\n")
            with pytest.raises(ApplyConflictError):
                await diff.apply(receipt)
            assert (environment.host_root / "project" / "input.txt").read_text() == "host changed\n"
        finally:
            await sandbox.discard()

    _run(scenario())


def test_diff_bundle_round_trip_and_digest(tmp_path: Path) -> None:
    async def scenario() -> None:
        sandbox, environment = await _open_fake(tmp_path)
        try:
            (environment.work_root / "project" / "new.txt").write_text("payload")
            diff = await environment.diff_against_host()
            path = tmp_path / "change.diff.zip"
            diff.write(path)
            loaded = DiffBundle.read(path)
            assert loaded.digest == diff.digest
            assert loaded.to_dict() == diff.to_dict()
        finally:
            await sandbox.discard()

    _run(scenario())


def test_snapshot_rejects_escape_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    link = root / "escape"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this Windows runner")
    with pytest.raises(SandboxSecurityError):
        capture_manifest(root)


def test_cli_spawn_and_diff_use_persisted_session(tmp_path: Path, capsys) -> None:
    from synapsekit.cli.main import main

    base = tmp_path / "host"
    base.mkdir()
    (base / "file.txt").write_text("before")
    state_dir = tmp_path / "sessions"
    main(
        [
            "sandbox",
            "spawn",
            "--backend",
            "fake",
            "--base",
            str(base),
            "--state-dir",
            str(state_dir),
            "--format",
            "json",
        ]
    )
    session = json.loads(capsys.readouterr().out)["session_id"]
    metadata = json.loads((state_dir / session / "metadata.json").read_text())
    Path(metadata["work_root"], "file.txt").write_text("changed")
    main(["sandbox", "diff", session, "--state-dir", str(state_dir), "--format", "json"])
    result = json.loads(capsys.readouterr().out)
    assert result["preview"]["modifications"] == 1
