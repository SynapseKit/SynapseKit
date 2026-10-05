"""Backend contracts and agent/environment integration tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from synapsekit.computer_use.agent import ComputerUseAgent
from synapsekit.computer_use.types import ComputerAction, ComputerActionType, ComputerObservation
from synapsekit.sandbox.backends.base import build_backend, run_process
from synapsekit.sandbox.backends.docker import DockerBackend
from synapsekit.sandbox.backends.fake import FakeBackend
from synapsekit.sandbox.backends.firecracker import FirecrackerBackend
from synapsekit.sandbox.backends.lima import LimaBackend
from synapsekit.sandbox.backends.podman import PodmanBackend
from synapsekit.sandbox.diff import DiffBundle
from synapsekit.sandbox.types import FileChange, FileChangeKind, SandboxConfig


def test_fake_backend_executes_in_the_worktree(tmp_path) -> None:
    async def scenario() -> None:
        work = tmp_path / "work"
        work.mkdir()
        backend = FakeBackend()
        handle = await backend.start(session_id="test", work_root=str(work), config=SandboxConfig())
        result = await backend.exec(
            handle,
            ["python", "-c", "from pathlib import Path; Path('created.txt').write_text('ok')"],
            timeout=10,
        )
        assert result.ok
        assert (work / "created.txt").read_text() == "ok"

    asyncio.run(scenario())


def test_docker_start_has_restrictive_defaults(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    async def fake_run(command, **kwargs):
        calls.append(list(command))
        from synapsekit.sandbox.types import CommandResult

        if command[:2] == ["docker", "version"]:
            return CommandResult(0, "27.0\n", "")
        return CommandResult(0, "container-id\n", "")

    monkeypatch.setattr("synapsekit.sandbox.backends.docker.run_process", fake_run)

    async def scenario() -> None:
        backend = DockerBackend()
        handle = await backend.start(
            session_id="abcdef1234567890",
            work_root=str(tmp_path),
            config=SandboxConfig(network="none"),
        )
        assert handle.identifier == "container-id"

    asyncio.run(scenario())
    command = calls[1]
    assert "--read-only" in command
    assert "--cap-drop" in command
    assert "ALL" in command
    assert "--security-opt" in command
    assert "no-new-privileges" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--privileged" not in command


def test_podman_start_has_restrictive_defaults(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    async def fake_run(command, **kwargs):
        calls.append(list(command))
        from synapsekit.sandbox.types import CommandResult

        if command[:2] == ["podman", "version"]:
            return CommandResult(0, "4.9.0\n", "")
        return CommandResult(0, "container-id\n", "")

    monkeypatch.setattr("synapsekit.sandbox.backends.podman.run_process", fake_run)

    async def scenario() -> None:
        backend = PodmanBackend()
        handle = await backend.start(
            session_id="abcdef1234567890",
            work_root=str(tmp_path),
            config=SandboxConfig(network="none"),
        )
        assert handle.identifier == "container-id"

        await backend.exec(handle, ["echo", "hi"], timeout=10)
        await backend.close(handle)

    asyncio.run(scenario())
    start_command = calls[1]
    assert "--read-only" in start_command
    assert "--cap-drop" in start_command
    assert "ALL" in start_command
    assert "--security-opt" in start_command
    assert "no-new-privileges" in start_command
    assert start_command[start_command.index("--network") + 1] == "none"
    assert "--privileged" not in start_command

    exec_command = calls[2]
    assert exec_command[:2] == ["podman", "exec"]
    assert exec_command[-2:] == ["echo", "hi"]

    close_command = calls[3]
    assert close_command == ["podman", "rm", "--force", "container-id"]


def test_podman_probe_reports_unavailable_backend(monkeypatch) -> None:
    async def fake_run(command, **kwargs):
        from synapsekit.sandbox.types import CommandResult

        return CommandResult(1, "", "podman: command not found")

    monkeypatch.setattr("synapsekit.sandbox.backends.podman.run_process", fake_run)

    capabilities = asyncio.run(PodmanBackend().probe())
    assert not capabilities.available
    assert "podman: command not found" in capabilities.reason


def test_build_backend_resolves_podman() -> None:
    assert isinstance(build_backend("podman"), PodmanBackend)


def test_vm_backends_fail_closed_on_unsupported_configuration(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")

    async def scenario() -> None:
        lima = await LimaBackend().probe()
        firecracker = await FirecrackerBackend().probe()
        assert not lima.available
        assert not firecracker.available

    asyncio.run(scenario())


def test_computer_use_agent_uses_environment_screen_without_closing_it() -> None:
    class Screen:
        closed = False

        async def observe(self):
            return ComputerObservation(text="ready")

        async def execute(self, action):
            return "ok"

        async def close(self):
            self.closed = True

    class Provider:
        async def next_action(self, task, observation, history):
            return ComputerAction(type=ComputerActionType.DONE, reason="done")

    screen = Screen()
    environment = SimpleNamespace(screen=screen)

    async def scenario() -> None:
        agent = ComputerUseAgent(provider=Provider(), screen=Screen())
        result = await agent.run("check", env=environment)
        assert result.completed

    asyncio.run(scenario())
    assert screen.closed is False


def test_computer_use_audit_records_text_length_not_secret_text() -> None:
    secret_typed = "hunter2-TYPED-secret"
    secret_screen = "on-screen-SECRET-abc"

    class Screen:
        async def observe(self):
            return ComputerObservation(text=secret_screen, app="editor")

        async def execute(self, action):
            return "typed"

        async def close(self):
            pass

    class Provider:
        def __init__(self):
            self.calls = 0

        async def next_action(self, task, observation, history):
            self.calls += 1
            if self.calls == 1:
                return ComputerAction(type=ComputerActionType.TYPE_TEXT, text=secret_typed)
            return ComputerAction(type=ComputerActionType.DONE, reason="done")

    class RecordingTracer:
        def __init__(self):
            self.payloads: list[dict] = []

        def record(self, kind, payload, *, actor=None):
            self.payloads.append(payload)

    tracer = RecordingTracer()
    environment = SimpleNamespace(screen=Screen(), tracer=tracer, session_id="s1")

    async def scenario() -> None:
        agent = ComputerUseAgent(provider=Provider(), screen=Screen())
        await agent.run("edit a file", env=environment)

    asyncio.run(scenario())

    blob = json.dumps(tracer.payloads)
    assert secret_typed not in blob
    assert secret_screen not in blob
    assert any(
        payload.get("action", {}).get("text_length") == len(secret_typed)
        for payload in tracer.payloads
    )
    assert any(
        payload.get("event") == "computer.observe"
        and payload.get("text_length") == len(secret_screen)
        for payload in tracer.payloads
    )


def test_apply_rolls_back_when_a_later_operation_is_invalid(tmp_path) -> None:
    root = tmp_path / "host"
    root.mkdir()
    bundle = DiffBundle(
        host_root=str(root),
        base_fingerprint="base",
        sandbox_id="sandbox",
        changes=(
            FileChange(FileChangeKind.ADD, "one.txt", payload=b"one"),
            FileChange(FileChangeKind.ADD, "two.txt", payload=None),
        ),
    )
    receipt = SimpleNamespace(passed=True, diff_sha256=bundle.digest)

    async def scenario() -> None:
        try:
            await bundle.apply(receipt)
        except Exception as exc:
            assert "missing payload" in str(exc)
        else:
            raise AssertionError("invalid operation unexpectedly applied")

    asyncio.run(scenario())
    assert not (root / "one.txt").exists()


def test_run_process_bounds_unbounded_output_and_truncates() -> None:
    async def scenario() -> None:
        cmd = [
            "python",
            "-c",
            "import sys, time; sys.stdout.write('x' * 10000); sys.stdout.flush(); time.sleep(10)",
        ]
        result = await run_process(cmd, timeout=5.0, max_output_bytes=100)
        assert not result.ok
        assert len(result.stdout) <= 100 + len("\n[output truncated]")
        assert "[output truncated]" in result.stdout

    asyncio.run(scenario())


def test_run_process_validates_max_output_bytes() -> None:
    async def scenario() -> None:
        with pytest.raises(ValueError, match="max_output_bytes must be positive"):
            await run_process(["python", "-c", "print(1)"], max_output_bytes=0)

    asyncio.run(scenario())


def test_sandbox_config_validates_max_output_bytes() -> None:
    config = SandboxConfig()
    assert config.max_output_bytes == 1_000_000
    with pytest.raises(ValueError, match="Sandbox max_output_bytes must be positive"):
        SandboxConfig(max_output_bytes=0)


def test_fake_backend_respects_max_output_bytes_from_config(tmp_path) -> None:
    async def scenario() -> None:
        work = tmp_path / "work"
        work.mkdir()
        backend = FakeBackend()
        config = SandboxConfig(max_output_bytes=50)
        handle = await backend.start(session_id="test-bytes", work_root=str(work), config=config)
        result = await backend.exec(
            handle,
            ["python", "-c", "import sys; sys.stdout.write('a' * 500); sys.stdout.flush()"],
            timeout=10,
            max_output_bytes=config.max_output_bytes,
        )
        assert "[output truncated]" in result.stdout
        assert len(result.stdout) <= 50 + len("\n[output truncated]")

    asyncio.run(scenario())
