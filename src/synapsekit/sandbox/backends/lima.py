"""Lima backend capability boundary.

Lima's VM image and mount policy are host-specific.  The adapter intentionally
fails closed until a configured instance is supplied rather than silently
running commands on the host.
"""

from __future__ import annotations

import platform
from collections.abc import Sequence

from ..errors import BackendUnavailableError
from ..types import BackendCapabilities, CommandResult, SandboxConfig
from .base import BackendHandle, SandboxBackend, run_process


class LimaBackend(SandboxBackend):
    name = "lima"

    async def probe(self) -> BackendCapabilities:
        if platform.system().lower() not in {"darwin", "linux"}:
            return BackendCapabilities(
                name=self.name, available=False, reason="Lima requires macOS or Linux."
            )
        result = await run_process(["limactl", "--version"], timeout=20.0)
        return BackendCapabilities(
            name=self.name,
            available=result.ok,
            native_cow=False,
            reason=result.stdout.strip()
            if result.ok
            else result.stderr.strip() or "limactl is unavailable.",
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
            or "Lima requires an explicitly configured VM image and read-only mount policy."
        )

    async def exec(
        self,
        handle: BackendHandle,
        command: Sequence[str],
        *,
        timeout: float,
    ) -> CommandResult:
        raise BackendUnavailableError("Lima execution is not configured for this host.")

    async def close(self, handle: BackendHandle) -> None:
        return None
