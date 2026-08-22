"""Backend protocol and factory."""

from __future__ import annotations

import asyncio
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..errors import BackendUnavailableError
from ..types import BackendCapabilities, CommandResult, SandboxConfig


@dataclass(slots=True)
class BackendHandle:
    """Persistable handle for a running backend instance."""

    backend: str
    identifier: str
    work_root: str
    metadata: dict[str, str] = field(default_factory=dict)


class SandboxBackend:
    """Async execution contract implemented by each backend."""

    name = "backend"

    async def probe(self) -> BackendCapabilities:
        raise NotImplementedError

    async def start(
        self,
        *,
        session_id: str,
        work_root: str,
        config: SandboxConfig,
    ) -> BackendHandle:
        raise NotImplementedError

    async def exec(
        self,
        handle: BackendHandle,
        command: Sequence[str],
        *,
        timeout: float,
    ) -> CommandResult:
        raise NotImplementedError

    async def close(self, handle: BackendHandle) -> None:
        raise NotImplementedError


async def run_process(
    command: Sequence[str],
    *,
    cwd: str | None = None,
    timeout: float = 120.0,
) -> CommandResult:
    """Run an executable without invoking a shell."""

    if not command:
        raise ValueError("Backend command must not be empty.")
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return CommandResult(returncode=127, stderr=f"Executable not found: {command[0]}")

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.communicate()
        return CommandResult(returncode=124, stderr=f"Command timed out after {timeout:.1f}s.")
    return CommandResult(
        returncode=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


def unavailable(capabilities: BackendCapabilities) -> BackendCapabilities:
    return capabilities


def build_backend(name: str) -> SandboxBackend:
    """Construct a backend by stable public name."""

    normalized = name.strip().lower()
    if normalized == "docker":
        from .docker import DockerBackend

        return DockerBackend()
    if normalized == "orbstack":
        from .docker import OrbStackBackend

        return OrbStackBackend()
    if normalized == "lima":
        from .lima import LimaBackend

        return LimaBackend()
    if normalized == "firecracker":
        from .firecracker import FirecrackerBackend

        return FirecrackerBackend()
    if normalized == "fake":
        from .fake import FakeBackend

        return FakeBackend()
    raise BackendUnavailableError(f"Unknown sandbox backend: {name!r}.")


def platform_is_linux() -> bool:
    return platform.system().lower() == "linux"


def platform_is_macos() -> bool:
    return platform.system().lower() == "darwin"
