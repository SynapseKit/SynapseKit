"""Backend contracts and agent/environment integration tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from synapsekit.computer_use.agent import ComputerUseAgent
from synapsekit.computer_use.types import ComputerAction, ComputerActionType, ComputerObservation
from synapsekit.sandbox.backends.docker import DockerBackend
from synapsekit.sandbox.backends.fake import FakeBackend
from synapsekit.sandbox.backends.firecracker import FirecrackerBackend
from synapsekit.sandbox.backends.lima import LimaBackend
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
