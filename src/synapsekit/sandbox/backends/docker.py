"""Docker and OrbStack-compatible sandbox backend."""

from __future__ import annotations

import platform
from collections.abc import Sequence

from ..errors import BackendUnavailableError
from ..types import BackendCapabilities, CommandResult, NetworkPolicy, SandboxConfig
from .base import BackendHandle, SandboxBackend, run_process


class DockerBackend(SandboxBackend):
    """Run a session in a restricted, detached Docker container."""

    name = "docker"

    async def probe(self) -> BackendCapabilities:
        result = await run_process(["docker", "version", "--format", "{{.Server.Version}}"])
        if not result.ok:
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason=result.stderr.strip() or "Docker daemon is unavailable.",
            )
        return BackendCapabilities(
            name=self.name,
            available=True,
            native_cow=False,
            gui=False,
            network_policies=(NetworkPolicy.NONE.value, NetworkPolicy.EGRESS_ONLY.value),
            reason=result.stdout.strip() or None,
        )

    async def start(
        self,
        *,
        session_id: str,
        work_root: str,
        config: SandboxConfig,
    ) -> BackendHandle:
        capabilities = await self.probe()
        if not capabilities.available:
            raise BackendUnavailableError(capabilities.reason or "Docker is unavailable.")

        container_name = f"synapsekit-sandbox-{session_id[:20]}"
        command: list[str] = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            container_name,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(config.pids_limit),
            "--volume",
            f"{work_root}:/workspace:rw",
            "--workdir",
            "/workspace",
        ]
        if config.memory:
            command.extend(["--memory", config.memory])
        if config.cpus is not None:
            command.extend(["--cpus", str(config.cpus)])
        if config.network == NetworkPolicy.NONE.value:
            command.extend(["--network", "none"])
        elif config.network == NetworkPolicy.EGRESS_ONLY.value:
            command.extend(["--network", "bridge"])
        for key, value in config.sanitized_environment.items():
            command.extend(["--env", f"{key}={value}"])
        command.extend([config.image, "sleep", "infinity"])

        result = await run_process(command, timeout=config.command_timeout)
        if not result.ok:
            raise BackendUnavailableError(
                result.stderr.strip() or "Docker failed to start sandbox."
            )
        identifier = result.stdout.strip().splitlines()[-1]
        return BackendHandle(
            backend=self.name,
            identifier=identifier,
            work_root=work_root,
            metadata={"container_name": container_name},
        )

    async def exec(
        self,
        handle: BackendHandle,
        command: Sequence[str],
        *,
        timeout: float,
    ) -> CommandResult:
        return await run_process(
            ["docker", "exec", "--workdir", "/workspace", handle.identifier, *command],
            timeout=timeout,
        )

    async def close(self, handle: BackendHandle) -> None:
        await run_process(["docker", "rm", "--force", handle.identifier], timeout=30.0)


class OrbStackBackend(DockerBackend):
    """OrbStack adapter using its Docker-compatible local engine on macOS."""

    name = "orbstack"

    async def probe(self) -> BackendCapabilities:
        if platform.system().lower() != "darwin":
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason="OrbStack is supported only on macOS.",
            )
        capabilities = await super().probe()
        return BackendCapabilities(
            name=self.name,
            available=capabilities.available,
            native_cow=capabilities.native_cow,
            gui=bool(capabilities.available),
            network_policies=capabilities.network_policies,
            reason=capabilities.reason,
        )
