"""Backend protocol and factory."""

from __future__ import annotations

import asyncio
import contextlib
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
        max_output_bytes: int = 1_000_000,
    ) -> CommandResult:
        raise NotImplementedError

    async def close(self, handle: BackendHandle) -> None:
        raise NotImplementedError


async def _drain_process(
    process: asyncio.subprocess.Process,
    limit: int,
) -> tuple[bytes, bytes, bool]:
    overflowed = False

    async def _read(stream: asyncio.StreamReader | None) -> bytes:
        nonlocal overflowed
        if stream is None:
            return b""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                overflowed = True
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                break
        return b"".join(chunks)

    stdout, stderr = await asyncio.gather(
        _read(process.stdout),
        _read(process.stderr),
    )
    await process.wait()
    return stdout, stderr, overflowed


def _decode_output(value: bytes | None, limit: int) -> str:
    if not value:
        return ""
    clipped = value[:limit]
    suffix = "\n[output truncated]" if len(value) > limit else ""
    return clipped.decode("utf-8", errors="replace") + suffix


async def run_process(
    command: Sequence[str],
    *,
    cwd: str | None = None,
    timeout: float = 120.0,
    max_output_bytes: int = 1_000_000,
) -> CommandResult:
    """Run an executable without invoking a shell."""

    if not command:
        raise ValueError("Backend command must not be empty.")
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive.")
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
        stdout, stderr, _ = await asyncio.wait_for(
            _drain_process(process, max_output_bytes), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return CommandResult(returncode=124, stderr=f"Command timed out after {timeout:.1f}s.")
    return CommandResult(
        returncode=process.returncode if process.returncode is not None else 0,
        stdout=_decode_output(stdout, max_output_bytes),
        stderr=_decode_output(stderr, max_output_bytes),
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
    if normalized == "podman":
        from .podman import PodmanBackend

        return PodmanBackend()
    raise BackendUnavailableError(f"Unknown sandbox backend: {name!r}.")


def platform_is_linux() -> bool:
    return platform.system().lower() == "linux"


def platform_is_macos() -> bool:
    return platform.system().lower() == "darwin"
