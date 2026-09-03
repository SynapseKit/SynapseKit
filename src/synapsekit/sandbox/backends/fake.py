"""Deterministic local backend used by tests and examples."""

from __future__ import annotations

from collections.abc import Sequence

from ..types import BackendCapabilities, CommandResult, NetworkPolicy, SandboxConfig
from .base import BackendHandle, SandboxBackend, run_process


class FakeBackend(SandboxBackend):
    """Execute commands in a materialized test worktree.

    This backend is intentionally not a security boundary and must never be
    selected as a production backend.
    """

    name = "fake"

    async def probe(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=self.name,
            available=True,
            native_cow=False,
            gui=False,
            network_policies=(NetworkPolicy.NONE.value,),
        )

    async def start(
        self,
        *,
        session_id: str,
        work_root: str,
        config: SandboxConfig,
    ) -> BackendHandle:
        return BackendHandle(backend=self.name, identifier=session_id, work_root=work_root)

    async def exec(
        self,
        handle: BackendHandle,
        command: Sequence[str],
        *,
        timeout: float,
    ) -> CommandResult:
        return await run_process(command, cwd=handle.work_root, timeout=timeout)

    async def close(self, handle: BackendHandle) -> None:
        return None
