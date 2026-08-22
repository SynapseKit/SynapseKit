"""PC Twin lifecycle orchestration."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ..audit import AuditTracer, EventKind
from .backends import BackendHandle, build_backend
from .environment import SandboxEnvironment
from .errors import SandboxSecurityError, SandboxStateError
from .overlay import FsOverlay
from .types import SandboxConfig, SandboxState, SnapshotManifest, normalize_relative_path


class PCSandbox:
    """Create and manage an ephemeral, auditable PC Twin session."""

    def __init__(
        self,
        *,
        base: str | Path = "current_user",
        backend: str = "docker",
        network: str = "none",
        fs_overlay: bool = True,
        include: tuple[str, ...] | list[str] = (),
        exclude: tuple[str, ...] | list[str] | None = None,
        state_dir: str | Path | None = None,
        image: str = "python:3.12-slim",
        command_timeout: float = 120.0,
        memory: str | None = "1g",
        cpus: float | None = 2.0,
        pids_limit: int = 256,
        environment: dict[str, str] | None = None,
        screen: Any | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "base": base,
            "backend": backend,
            "network": network,
            "fs_overlay": fs_overlay,
            "include": tuple(include),
            "state_dir": None if state_dir is None else Path(state_dir),
            "image": image,
            "command_timeout": command_timeout,
            "memory": memory,
            "cpus": cpus,
            "pids_limit": pids_limit,
            "environment": dict(environment or {}),
        }
        if exclude is not None:
            values["exclude"] = tuple(exclude)
        self.config = SandboxConfig(**values)
        self.screen = screen
        self.session_id: str | None = None
        self._environment: SandboxEnvironment | None = None

    async def __aenter__(self) -> PCSandbox:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.discard()

    @property
    def environment(self) -> SandboxEnvironment:
        """Return the active environment, or fail if the session is unopened."""

        if self._environment is None:
            raise SandboxStateError("Sandbox has not been started.")
        return self._environment

    @asynccontextmanager
    async def snapshot(self) -> AsyncIterator[SandboxEnvironment]:
        """Open a snapshot and discard it automatically unless already applied."""

        environment = await self.start()
        try:
            yield environment
        finally:
            if environment.state not in {SandboxState.APPLIED, SandboxState.DISCARDED}:
                await environment.discard()

    async def start(self) -> SandboxEnvironment:
        if self._environment is not None:
            return self._environment

        base = self.config.resolved_base
        if not base.is_dir():
            raise FileNotFoundError(f"Sandbox base directory does not exist: {base}")
        state_dir = self.config.resolved_state_dir
        await asyncio.to_thread(state_dir.mkdir, parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex
        session_root = state_dir / session_id
        work_root = session_root / "workspace"
        await asyncio.to_thread(session_root.mkdir, parents=True, exist_ok=False)
        tracer = AuditTracer(run_id=f"sandbox-{session_id}")
        tracer.record(
            EventKind.SYSTEM_EVENT,
            {"event": "sandbox.prepare", "base": str(base), "backend": self.config.backend},
            actor=f"sandbox:{session_id}",
        )

        excludes = list(self.config.exclude)
        try:
            session_relative = session_root.relative_to(base).as_posix()
        except ValueError:
            session_relative = ""
        if session_relative:
            excludes.append(session_relative)

        try:
            overlay = FsOverlay(
                base,
                include=self.config.include,
                exclude=tuple(excludes),
            )
            baseline = await asyncio.to_thread(overlay.materialize, work_root)
            backend = build_backend(self.config.backend)
            handle = await backend.start(
                session_id=session_id,
                work_root=str(work_root),
                config=self.config,
            )
        except Exception:
            shutil.rmtree(session_root, ignore_errors=True)
            raise

        environment = SandboxEnvironment(
            session_id=session_id,
            host_root=base,
            work_root=work_root,
            baseline=baseline,
            config=self.config,
            backend=backend,
            handle=handle,
            tracer=tracer,
            screen=self.screen,
            _cleanup=lambda: self._cleanup_session(session_root),
        )
        self.session_id = session_id
        self._environment = environment
        await asyncio.to_thread(self._write_metadata, environment, session_root)
        tracer.record(
            EventKind.SYSTEM_EVENT,
            {"event": "sandbox.running", "session_id": session_id},
            actor=f"sandbox:{session_id}",
        )
        return environment

    async def diff(self):
        """Generate a reviewable diff for the active session."""

        return await self.environment.diff_against_host()

    async def evaluate(self, gate: Any, diff: Any | None = None):
        """Evaluate the active session through an explicit gate."""

        return await self.environment.evaluate(gate, diff)

    async def apply(self, diff: Any, receipt: Any) -> None:
        """Apply a passing diff to the host through the active environment."""

        await self.environment.apply(diff, receipt)

    async def run(
        self,
        agent_path: Path,
        manifest: Any,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        """Run an installed agent through the marketplace ``AgentSandbox`` protocol.

        Agent files are copied to an internal overlay directory rather than
        mounted from the host. The entrypoint receives the prompt as a normal
        argv value after ``--prompt``; it is never interpolated into a shell.
        """

        python_executable = kwargs.pop("python_executable", "python")
        extra_args = kwargs.pop("args", ())
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported sandbox agent options: {unexpected}")
        if not isinstance(python_executable, str) or not python_executable:
            raise ValueError("python_executable must be a non-empty executable name.")
        if not isinstance(extra_args, Sequence) or isinstance(extra_args, (str, bytes)):
            raise ValueError("args must be a sequence of argument strings.")
        if any(not isinstance(value, str) for value in extra_args):
            raise ValueError("args must contain only strings.")

        environment = await self.start()
        entrypoint = getattr(manifest, "entrypoint", None)
        if not isinstance(entrypoint, str) or not entrypoint:
            raise SandboxSecurityError("Agent manifest must define a relative entrypoint.")
        staged_entrypoint = await asyncio.to_thread(
            self._stage_agent,
            Path(agent_path),
            entrypoint,
            environment.work_root,
        )
        relative_entrypoint = staged_entrypoint.relative_to(environment.work_root).as_posix()
        result = await environment.exec(
            (
                python_executable,
                "-I",
                relative_entrypoint,
                "--prompt",
                prompt,
                *extra_args,
            )
        )
        environment.tracer.record(
            EventKind.SYSTEM_EVENT,
            {
                "event": "sandbox.agent.run",
                "agent": getattr(manifest, "name", None),
                "entrypoint": entrypoint,
                "returncode": result.returncode,
            },
            actor=f"sandbox:{environment.session_id}",
        )
        return result

    @staticmethod
    def _stage_agent(agent_path: Path, entrypoint: str, work_root: Path) -> Path:
        source = agent_path.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Installed agent directory does not exist: {source}")
        safe_entrypoint = normalize_relative_path(entrypoint)
        entry = source.joinpath(*safe_entrypoint.split("/"))
        try:
            entry.resolve(strict=True).relative_to(source)
        except ValueError as exc:
            raise SandboxSecurityError("Agent entrypoint escapes the installed bundle.") from exc
        if not entry.is_file() or entry.is_symlink():
            raise SandboxSecurityError("Agent entrypoint must be a regular file.")

        destination = work_root / ".synapsekit" / "agents" / uuid.uuid4().hex
        for item in sorted(source.rglob("*"), key=lambda path: path.as_posix()):
            relative = item.relative_to(source)
            target = destination / relative
            if item.is_symlink():
                raise SandboxSecurityError("Installed agent contains a symbolic link.")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
            else:
                raise SandboxSecurityError(f"Unsupported agent filesystem entry: {item}")
        return destination.joinpath(*safe_entrypoint.split("/"))

    async def discard(self) -> None:
        if self._environment is not None:
            await self._environment.discard()

    @staticmethod
    def _cleanup_session(session_root: Path) -> None:
        shutil.rmtree(session_root, ignore_errors=True)

    @staticmethod
    def _write_metadata(environment: SandboxEnvironment, session_root: Path) -> None:
        payload = {
            "session_id": environment.session_id,
            "state": environment.state.value,
            "host_root": str(environment.host_root),
            "work_root": str(environment.work_root),
            "backend": environment.handle.backend,
            "handle": {
                "identifier": environment.handle.identifier,
                "metadata": environment.handle.metadata,
            },
            "config": {
                "base": str(environment.config.base),
                "backend": environment.config.backend,
                "network": str(environment.config.network),
                "include": list(environment.config.include),
                "exclude": list(environment.config.exclude),
                "fs_overlay": environment.config.fs_overlay,
                "image": environment.config.image,
                "command_timeout": environment.config.command_timeout,
                "memory": environment.config.memory,
                "cpus": environment.config.cpus,
                "pids_limit": environment.config.pids_limit,
                "environment": environment.config.sanitized_environment,
            },
            "baseline": {
                "root": environment.baseline.root,
                "items": list(environment.baseline.items),
                "fingerprint": environment.baseline.fingerprint,
            },
        }
        (session_root / "metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    async def attach(cls, session_id: str, *, state_dir: str | Path | None = None) -> PCSandbox:
        """Reattach to a persisted session created by ``sandbox spawn``."""

        root = Path(state_dir or (Path.home() / ".synapsekit" / "sandboxes")) / session_id
        metadata = root / "metadata.json"
        payload = json.loads(await asyncio.to_thread(metadata.read_text, encoding="utf-8"))
        config_data = payload["config"]
        sandbox = cls(
            base=config_data["base"],
            backend=config_data["backend"],
            network=config_data["network"],
            fs_overlay=bool(config_data["fs_overlay"]),
            include=tuple(config_data["include"]),
            exclude=tuple(config_data["exclude"]),
            state_dir=root.parent,
            image=config_data["image"],
            command_timeout=float(config_data["command_timeout"]),
            memory=config_data["memory"],
            cpus=config_data["cpus"],
            pids_limit=int(config_data["pids_limit"]),
            environment=dict(config_data["environment"]),
        )
        backend = build_backend(config_data["backend"])
        handle = BackendHandle(
            backend=payload["backend"],
            identifier=payload["handle"]["identifier"],
            work_root=payload["work_root"],
            metadata=dict(payload["handle"].get("metadata", {})),
        )
        baseline_data = payload["baseline"]
        environment = SandboxEnvironment(
            session_id=session_id,
            host_root=Path(payload["host_root"]),
            work_root=Path(payload["work_root"]),
            baseline=SnapshotManifest(
                root=baseline_data["root"],
                items=tuple(baseline_data["items"]),
                fingerprint=baseline_data["fingerprint"],
            ),
            config=sandbox.config,
            backend=backend,
            handle=handle,
            tracer=AuditTracer(run_id=f"sandbox-{session_id}"),
            _cleanup=lambda: cls._cleanup_session(root),
            state=SandboxState(payload.get("state", SandboxState.RUNNING.value)),
        )
        sandbox.session_id = session_id
        sandbox._environment = environment
        return sandbox
