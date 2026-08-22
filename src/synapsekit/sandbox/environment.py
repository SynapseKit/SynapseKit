"""Active sandbox environment exposed to agents and evaluation gates."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..audit import AuditTracer, EventKind
from .backends.base import BackendHandle, SandboxBackend
from .diff import DiffBundle
from .eval import EvalGate, EvalReceipt
from .overlay import capture_manifest
from .types import CommandResult, SandboxConfig, SandboxState, SnapshotManifest

if TYPE_CHECKING:
    from ..audit import SigningPolicy


@dataclass(slots=True)
class SandboxEnvironment:
    """A running, isolated working tree and its backend handle."""

    session_id: str
    host_root: Path
    work_root: Path
    baseline: SnapshotManifest
    config: SandboxConfig
    backend: SandboxBackend
    handle: BackendHandle
    tracer: AuditTracer
    screen: Any | None = None
    state: SandboxState = SandboxState.RUNNING
    _cleanup: Callable[[], Any] | None = None

    async def exec(self, command: Sequence[str]) -> CommandResult:
        if self.state not in {SandboxState.RUNNING, SandboxState.DIFFED, SandboxState.EVALUATED}:
            raise RuntimeError(f"Cannot execute commands in state {self.state.value}.")
        result = await self.backend.exec(
            self.handle,
            tuple(command),
            timeout=self.config.command_timeout,
        )
        self.tracer.record(
            EventKind.SYSTEM_EVENT,
            {
                "event": "sandbox.exec",
                "command": list(command),
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            },
            actor=f"sandbox:{self.session_id}",
        )
        return result

    async def diff_against_host(self) -> DiffBundle:
        if self.state not in {SandboxState.RUNNING, SandboxState.DIFFED, SandboxState.EVALUATED}:
            raise RuntimeError(f"Cannot diff sandbox in state {self.state.value}.")
        after = await asyncio.to_thread(
            capture_manifest,
            self.work_root,
            include=self.config.include,
            exclude=self.config.exclude,
        )
        diff = DiffBundle.from_manifests(
            self.baseline,
            after,
            current_root=self.work_root,
            sandbox_id=self.session_id,
            audit_run_id=self.tracer.run_id,
        )
        self.state = SandboxState.DIFFED
        self.tracer.record(
            EventKind.SYSTEM_EVENT,
            {
                "event": "sandbox.diff",
                "diff_sha256": diff.digest,
                "changes": diff.preview().changes,
            },
            actor=f"sandbox:{self.session_id}",
        )
        return diff

    async def evaluate(self, gate: EvalGate, diff: DiffBundle | None = None) -> EvalReceipt:
        active_diff = diff or await self.diff_against_host()
        receipt = await gate.evaluate(active_diff, self)
        self.state = SandboxState.EVALUATED
        self.tracer.record(
            EventKind.SYSTEM_EVENT,
            {
                "event": "sandbox.eval",
                "evaluator": receipt.evaluator,
                "passed": receipt.passed,
                "score": receipt.score,
                "diff_sha256": receipt.diff_sha256,
            },
            actor=f"sandbox:{self.session_id}",
        )
        return receipt

    async def apply(self, diff: DiffBundle, receipt: EvalReceipt) -> None:
        """Apply a passed bundle to the original host root and record the transition."""

        await diff.apply(receipt, host_root=self.host_root)
        await self.mark_applied()
        self.tracer.record(
            EventKind.SYSTEM_EVENT,
            {
                "event": "sandbox.apply",
                "diff_sha256": diff.digest,
                "changes": diff.preview().changes,
            },
            actor=f"sandbox:{self.session_id}",
        )

    def export_audit_bundle(self, output_path: str | Path, signing_policy: SigningPolicy) -> Path:
        """Write the session's hash-chained trace as a signed portable bundle."""

        from ..audit import export_audit_bundle

        destination = Path(output_path)
        self.tracer.record(
            EventKind.SYSTEM_EVENT,
            {"event": "sandbox.audit.export", "output_path": str(destination)},
            actor=f"sandbox:{self.session_id}",
        )
        return Path(export_audit_bundle(list(self.tracer.records), signing_policy, destination))

    async def discard(self) -> None:
        if self.state in {SandboxState.DISCARDED, SandboxState.CLOSED}:
            return
        await self.backend.close(self.handle)
        if self._cleanup is not None:
            result = await asyncio.to_thread(self._cleanup)
            if hasattr(result, "__await__"):
                await result
        else:
            await asyncio.to_thread(shutil.rmtree, self.work_root.parent, True)
        self.state = SandboxState.DISCARDED

    async def mark_applied(self) -> None:
        self.state = SandboxState.APPLIED
