"""Tiny JSON status file for the ambient daemon (one row, no store needed)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_STATUS_PATH = Path.home() / ".synapsekit" / "ambient.status.json"


@dataclass
class AmbientStatus:
    state: str = "stopped"
    pid: int | None = None
    started_at: str | None = None


def read_status(path: str | Path = DEFAULT_STATUS_PATH) -> AmbientStatus:
    status_path = Path(path).expanduser()
    if not status_path.exists():
        return AmbientStatus()
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AmbientStatus()
    return AmbientStatus(
        state=data.get("state", "stopped"),
        pid=data.get("pid"),
        started_at=data.get("started_at"),
    )


def write_status(path: str | Path = DEFAULT_STATUS_PATH, **values: object) -> AmbientStatus:
    status_path = Path(path).expanduser()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = read_status(status_path)
    for key, value in values.items():
        setattr(status, key, value)
    status_path.write_text(json.dumps(asdict(status)), encoding="utf-8")
    return status
