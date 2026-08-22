"""Firecracker capability boundary."""

from __future__ import annotations

import platform
from collections.abc import Sequence

from ..errors import BackendUnavailableError
from ..types import BackendCapabilities, CommandResult, SandboxConfig
from .base import BackendHandle, SandboxBackend, run_process


class FirecrackerBackend(SandboxBackend):
    name = "firecracker"

    async def probe(self) -> BackendCapabilities:
        if platform.system().lower() != "linux":
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason="Firecracker requires Linux.",
            )
        result = await run_process(["firecracker", "--version"], timeout=20.0)
        return BackendCapabilities(
            name=self.name,
            available=result.ok,
            native_cow=True,
            reason=result.stdout.strip()
            if result.ok
            else result.stderr.strip() or "firecracker is unavailable.",
        )

    async def start(
        self,
        *,
        session_id: str,
        work_root: str,
        config: SandboxConfig,
    ) -> BackendHandle:
        capabilities = await self.probe()
        raise BackendUnavailableError(
            capabilities.reason
            or "Firecracker requires provisioned kernel, rootfs, and device configuration."
        )

    async def exec(
        self,
        handle: BackendHandle,
        command: Sequence[str],
        *,
        timeout: float,
    ) -> CommandResult:
        raise BackendUnavailableError("Firecracker execution is not configured for this host.")

    async def close(self, handle: BackendHandle) -> None:
        return None
