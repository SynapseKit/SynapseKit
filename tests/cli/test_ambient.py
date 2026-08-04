"""Tests for ``synapsekit ambient`` CLI wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from synapsekit.ambient import AmbientDaemonConfig
from synapsekit.cli.ambient import run_ambient


def _args(command: str, **extra) -> argparse.Namespace:
    payload = {
        "ambient_command": command,
        "repo": None,
        "json": True,
        "poll_interval": 2.0,
        "min_confidence": 0.6,
        "limit": 20,
    }
    payload.update(extra)
    return argparse.Namespace(**payload)


def _patch_config(monkeypatch, tmp_path: Path, *, audit_path: Path | None = None) -> None:
    config = AmbientDaemonConfig(
        status_path=tmp_path / "status.json",
        audit_path=audit_path or (tmp_path / "audit.jsonl"),
    )
    monkeypatch.setattr("synapsekit.cli.ambient._ambient_config", lambda args: config)


def test_ambient_cli_status_when_never_started(tmp_path: Path, capsys, monkeypatch) -> None:
    _patch_config(monkeypatch, tmp_path)

    run_ambient(_args("status"))
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "stopped"


def test_ambient_cli_missing_action_raises() -> None:
    with pytest.raises(SystemExit):
        run_ambient(_args(None))


def test_ambient_cli_log_reads_audit_entries(tmp_path: Path, capsys, monkeypatch) -> None:
    from synapsekit.observability import AuditLog

    audit_path = tmp_path / "audit.jsonl"
    AuditLog(backend="jsonl", path=str(audit_path)).record(
        model="destructive-delete", input_text="rm -rf build", output_text="careful!"
    )

    _patch_config(monkeypatch, tmp_path, audit_path=audit_path)

    run_ambient(_args("log"))
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["model"] == "destructive-delete"
