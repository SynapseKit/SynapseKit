"""Public data contracts for the sandbox package."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SandboxState(str, Enum):
    """Persisted lifecycle state for a sandbox session."""

    NEW = "new"
    PREPARING = "preparing"
    RUNNING = "running"
    DIFFED = "diffed"
    EVALUATED = "evaluated"
    APPLIED = "applied"
    DISCARDED = "discarded"
    CLOSED = "closed"
    FAILED = "failed"


class NetworkPolicy(str, Enum):
    """Network policy exposed to a sandbox backend."""

    NONE = "none"
    EGRESS_ONLY = "egress_only"


class FileChangeKind(str, Enum):
    """Filesystem operation represented by a diff bundle."""

    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    MKDIR = "mkdir"
    RMDIR = "rmdir"


@dataclass(frozen=True, slots=True)
class FileChange:
    """One safe, relative filesystem operation."""

    kind: FileChangeKind
    path: str
    before_sha256: str | None = None
    after_sha256: str | None = None
    size: int = 0
    mode: int | None = None
    payload: bytes | None = None


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Deterministic snapshot metadata for a selected host root."""

    root: str
    items: tuple[dict[str, Any], ...]
    fingerprint: str

    @property
    def file_count(self) -> int:
        return sum(item.get("kind") == "file" for item in self.items)

    @property
    def total_bytes(self) -> int:
        return sum(int(item.get("size", 0)) for item in self.items)


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Runtime capabilities reported by a backend probe."""

    name: str
    available: bool
    native_cow: bool = False
    gui: bool = False
    network_policies: tuple[str, ...] = (NetworkPolicy.NONE.value,)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured result of a command executed in the sandbox."""

    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


_DEFAULT_EXCLUDES = (
    ".synapsekit",
    ".ssh",
    ".aws",
    ".azure",
    ".config/gcloud",
    ".docker/config.json",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".env",
)


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Validated configuration shared by all sandbox backends."""

    base: str | Path = "current_user"
    backend: str = "docker"
    network: NetworkPolicy | str = NetworkPolicy.NONE
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = _DEFAULT_EXCLUDES
    fs_overlay: bool = True
    state_dir: Path | None = None
    image: str = "python:3.12-slim"
    command_timeout: float = 120.0
    memory: str | None = "1g"
    cpus: float | None = 2.0
    pids_limit: int = 256
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = self.backend.strip().lower()
        if not backend:
            raise ValueError("Sandbox backend must not be empty.")
        object.__setattr__(self, "backend", backend)
        network = (
            self.network.value if isinstance(self.network, NetworkPolicy) else str(self.network)
        )
        try:
            network = NetworkPolicy(network).value
        except ValueError as exc:
            raise ValueError(f"Unsupported sandbox network policy: {network!r}.") from exc
        object.__setattr__(self, "network", network)
        if self.command_timeout <= 0:
            raise ValueError("Sandbox command_timeout must be positive.")
        if self.cpus is not None and self.cpus <= 0:
            raise ValueError("Sandbox cpus must be positive when supplied.")
        if self.pids_limit <= 0:
            raise ValueError("Sandbox pids_limit must be positive.")
        if not self.image.strip():
            raise ValueError("Sandbox image must not be empty.")

    @property
    def resolved_base(self) -> Path:
        if self.base == "current_user":
            return Path.home().resolve()
        return Path(self.base).expanduser().resolve()

    @property
    def resolved_state_dir(self) -> Path:
        return (self.state_dir or (Path.home() / ".synapsekit" / "sandboxes")).expanduser()

    @property
    def sanitized_environment(self) -> dict[str, str]:
        return dict(self.environment)


def normalize_relative_path(path: str) -> str:
    """Normalize a bundle path and reject traversal/absolute paths."""

    value = path.replace("\\", "/")
    if not value or value.startswith("/") or "\x00" in value:
        raise ValueError(f"Unsafe relative path: {path!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe relative path: {path!r}")
    if os.path.splitdrive(value)[0]:
        raise ValueError(f"Unsafe relative path: {path!r}")
    return "/".join(parts)
