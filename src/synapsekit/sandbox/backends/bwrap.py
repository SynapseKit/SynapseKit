"""Bubblewrap (bwrap) unprivileged sandbox backend for Linux."""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..errors import BackendUnavailableError
from ..types import BackendCapabilities, CommandResult, NetworkPolicy, SandboxConfig
from .base import BackendHandle, SandboxBackend, platform_is_linux, run_process


class BubblewrapBackend(SandboxBackend):
    """Run an isolated sandbox using unprivileged Linux namespaces via Bubblewrap (bwrap).

    Bubblewrap allows running unprivileged, hermetic sandboxes without requiring root
    privileges, sudo, or background container daemons (like Docker or Podman). It leverages
    Linux kernel user namespaces, mount namespaces, and network namespaces directly.
    """

    name = "bubblewrap"

    async def probe(self) -> BackendCapabilities:
        if not platform_is_linux():
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason="Bubblewrap (bwrap) requires Linux.",
            )
        result = await run_process(["bwrap", "--version"])
        if not result.ok:
            return BackendCapabilities(
                name=self.name,
                available=False,
                reason=result.stderr.strip() or "Bubblewrap executable (bwrap) is unavailable.",
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
            raise BackendUnavailableError(capabilities.reason or "Bubblewrap is unavailable.")

        identifier = f"synapsekit-bwrap-{session_id[:20]}"
        metadata: dict[str, str] = {
            "network": config.network,
            "pids_limit": str(config.pids_limit),
        }
        if config.environment:
            metadata["environment"] = json.dumps(config.sanitized_environment)

        return BackendHandle(
            backend=self.name,
            identifier=identifier,
            work_root=work_root,
            metadata=metadata,
        )

    def _build_command(
        self,
        handle: BackendHandle,
        command: Sequence[str],
    ) -> list[str]:
        """Construct the full bwrap invocation with restrictive sandboxing flags."""
        bwrap_cmd: list[str] = [
            "bwrap",
            # Minimal read-only system rootfs. We deliberately do NOT bind the
            # host root ("--ro-bind / /") or the host home directory: the sandbox
            # is meant to see only the filtered workspace overlay, and binding the
            # real root would expose ~/.ssh, ~/.aws, ~/.env and every other host
            # secret (the exact files SandboxConfig.exclude strips from the copy).
            # Only the system directories needed to run interpreters/tools are
            # mounted; "--ro-bind-try" skips any that do not exist on this host.
            "--ro-bind-try",
            "/usr",
            "/usr",
            "--ro-bind-try",
            "/bin",
            "/bin",
            "--ro-bind-try",
            "/sbin",
            "/sbin",
            "--ro-bind-try",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--ro-bind-try",
            "/etc",
            "/etc",
            # Safe virtual filesystems
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            # Private writable scratch space
            "--tmpfs",
            "/tmp",
            # Isolated workspace mount
            "--dir",
            "/workspace",
            "--bind",
            handle.work_root,
            "/workspace",
            "--chdir",
            "/workspace",
            # Lifecycle and session isolation
            "--die-with-parent",
            "--new-session",
            # Namespace isolation
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
        ]

        network = handle.metadata.get("network", NetworkPolicy.NONE.value)
        if network == NetworkPolicy.NONE.value:
            bwrap_cmd.append("--unshare-net")

        # Environment sanitization
        bwrap_cmd.append("--clearenv")
        bwrap_cmd.extend(
            ["--setenv", "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"]
        )
        bwrap_cmd.extend(["--setenv", "HOME", "/workspace"])
        bwrap_cmd.extend(["--setenv", "TMPDIR", "/tmp"])

        env_raw = handle.metadata.get("environment")
        if env_raw:
            try:
                env_dict = json.loads(env_raw)
                for k, v in env_dict.items():
                    bwrap_cmd.extend(["--setenv", str(k), str(v)])
            except Exception:
                pass

        bwrap_cmd.extend(["--", *command])
        return bwrap_cmd

    async def exec(
        self,
        handle: BackendHandle,
        command: Sequence[str],
        *,
        timeout: float,
        max_output_bytes: int = 1_000_000,
    ) -> CommandResult:
        full_command = self._build_command(handle, command)
        return await run_process(
            full_command,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
        )

    async def close(self, handle: BackendHandle) -> None:
        """Bubblewrap sessions exit cleanly with their process; no daemon cleanup is required."""
        return None
