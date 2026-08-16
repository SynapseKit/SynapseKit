"""Daemon for the synapsekit ambient system."""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from synapsekit.mesh import MeshPrivacyFilter
from synapsekit.observability import AuditLog

from .._compat import run_sync
from .events import AmbientState
from .notify import notify_windows_toast
from .privacy import DEFAULT_AMBIENT_IGNORE, load_disabled_sources
from .rules import DEFAULT_MIN_CONFIDENCE, Intervention, evaluate
from .sources import GitSourcePlugin, TerminalSourcePlugin
from .sources.base import AmbientSourcePlugin
from .status import DEFAULT_STATUS_PATH, AmbientStatus, read_status, write_status

logger = logging.getLogger(__name__)

_STOP_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGINT)
_STOP_TIMEOUT_SECONDS = 5.0
_STOP_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class AmbientDaemonConfig:
    """Runtime options for ``AmbientDaemon``."""

    repo_root: Path = field(default_factory=Path.cwd)
    poll_interval: float = 2.0
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    privacy_file: str | Path | None = DEFAULT_AMBIENT_IGNORE
    status_path: str | Path = DEFAULT_STATUS_PATH
    audit_path: str | Path = field(
        default_factory=lambda: Path.home() / ".synapsekit" / "ambient_audit.jsonl"
    )
    terminal_history_path: str | Path | None = None


class AmbientDaemon:
    """Polls enabled ambient sources and fires notifications for risky commands."""

    def __init__(
        self,
        *,
        config: AmbientDaemonConfig | None = None,
        sources: list[AmbientSourcePlugin] | None = None,
    ) -> None:
        self.config = config or AmbientDaemonConfig()
        self._sources = sources if sources is not None else self._build_default_sources()
        self._state = AmbientState()
        self._stop_event = asyncio.Event()
        self._signal_handlers_installed = False
        self._audit_log: AuditLog | None = None
        # Command text often contains secrets (tokens, passwords). Redact
        # before it is persisted to the on-disk audit log.
        self._redactor = MeshPrivacyFilter(ignore_file=None)

    def _build_default_sources(self) -> list[AmbientSourcePlugin]:
        disabled = load_disabled_sources(self.config.privacy_file)
        sources: list[AmbientSourcePlugin] = []
        if "git" not in disabled:
            sources.append(GitSourcePlugin(self.config.repo_root))
        if "terminal" not in disabled:
            sources.append(TerminalSourcePlugin(self.config.terminal_history_path))
        return sources

    @property
    def audit_log(self) -> AuditLog:
        if self._audit_log is None:
            audit_path = Path(self.config.audit_path)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._audit_log = AuditLog(backend="jsonl", path=str(audit_path))
        return self._audit_log

    async def _write_status(self, **kwargs: object) -> AmbientStatus:
        """Offload the status-file write (mkdir + read + write) off the loop."""

        return await asyncio.to_thread(write_status, self.config.status_path, **kwargs)

    async def _read_status(self) -> AmbientStatus:
        """Offload the status-file read off the loop."""

        return await asyncio.to_thread(read_status, self.config.status_path)

    async def start(self) -> AmbientStatus:
        """Start the daemon: load sources, then poll until stopped."""

        names = [source.name for source in self._sources]
        print(f"synapsekit ambient: observing {', '.join(names) or '(no sources enabled)'}")
        for source in self._sources:
            await source.on_load()

        await self._write_status(
            state="running",
            pid=os.getpid(),
            started_at=datetime.now(UTC).isoformat(),
        )
        self._install_signal_handlers()
        try:
            await self._poll_loop()
        finally:
            self._remove_signal_handlers()
            for source in self._sources:
                await source.on_unload()
        return await self._read_status()

    def start_sync(self) -> AmbientStatus:
        """Sync wrapper for ``start``."""

        return run_sync(self.start())

    async def stop(self) -> AmbientStatus:
        """Stop the daemon (in-process, or by signaling the running PID)."""

        self._stop_event.set()

        status = await self._read_status()
        remote_pid = status.pid if (status.pid and status.pid != os.getpid()) else None
        if remote_pid is not None and self._signal_pid(remote_pid):
            await self._wait_for_stopped(remote_pid)

        status = await self._read_status()
        if status.state == "stopped":
            return status

        # The signaled process is still alive and did not update its own
        # status in time. Do NOT mark it stopped: that would orphan a live
        # daemon and discard the pid needed to signal it again. Report the
        # real state so a later `stop` can retry.
        if remote_pid is not None and _pid_alive(remote_pid):
            logger.warning(
                "ambient: pid %s did not stop within %.0fs; leaving status running",
                remote_pid,
                _STOP_TIMEOUT_SECONDS,
            )
            return status

        # In-process stop, or the process has exited — reflect stopped.
        return await self._write_status(state="stopped", pid=None)

    def stop_sync(self) -> AmbientStatus:
        """Sync wrapper for ``stop``."""

        return run_sync(self.stop())

    def status(self) -> AmbientStatus:
        """Return current daemon status."""

        return read_status(self.config.status_path)

    async def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            await self._tick()
            await asyncio.sleep(self.config.poll_interval)
        await self._write_status(state="stopped", pid=None)

    async def _tick(self) -> None:
        for source in self._sources:
            for event in await source.poll():
                if event.kind == "git_status":
                    self._state.apply_git_status(event)
                    continue
                intervention = evaluate(event, self._state)
                if (
                    intervention is not None
                    and intervention.confidence >= self.config.min_confidence
                ):
                    await self._fire(intervention)

    async def _fire(self, intervention: Intervention) -> None:
        # The toast backend and the jsonl audit write both block; keep them
        # off the event loop (async-first).
        await asyncio.to_thread(self._emit, intervention)

    def _emit(self, intervention: Intervention) -> None:
        notify_windows_toast("SynapseKit ambient", intervention.message)
        self.audit_log.record(
            model=intervention.rule,
            input_text=self._redactor.redact_text(intervention.event.text),
            output_text=intervention.message,
            user=_current_user(),
            metadata={"confidence": intervention.confidence, "source": intervention.event.source},
        )

    def _install_signal_handlers(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in _STOP_SIGNALS:
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, self._stop_event.set)
        self._signal_handlers_installed = True

    def _remove_signal_handlers(self) -> None:
        if not self._signal_handlers_installed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in _STOP_SIGNALS:
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.remove_signal_handler(sig)
        self._signal_handlers_installed = False

    @staticmethod
    def _signal_pid(pid: int) -> bool:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except PermissionError:
            return False
        return True

    async def _wait_for_stopped(self, pid: int) -> None:
        elapsed = 0.0
        while elapsed < _STOP_TIMEOUT_SECONDS:
            if (await self._read_status()).state == "stopped":
                return
            if not _pid_alive(pid):
                return
            await asyncio.sleep(_STOP_POLL_SECONDS)
            elapsed += _STOP_POLL_SECONDS


def _pid_alive(pid: int) -> bool:
    """Return whether ``pid`` refers to a live process (best-effort)."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"
