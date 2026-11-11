"""Reconnecting an MCPClient must close the connection it replaces."""

from __future__ import annotations

import asyncio
import sys
import types
import warnings
from typing import Any

import pytest

from synapsekit.mcp.client import MCPClient


class _Recorder:
    """Records enter/exit of every fake transport and session, in order."""

    def __init__(self) -> None:
        self.events: list[str] = []


def _install_fake_mcp(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    counter = {"n": 0}

    class _TaskBound:
        """A fake anyio-style context manager that is bound to one task.

        The real MCP transports / ``ClientSession`` are built on anyio cancel
        scopes, so ``__aexit__`` must run on the task that entered them. These
        fakes reproduce that: exiting on a different task raises, exactly as
        the SDK does -- which is what #1099 was about.
        """

        name: str

        async def __aenter__(self) -> Any:
            self._enter_task = asyncio.current_task()
            recorder.events.append(f"enter {self.name}")
            return self._opened()

        async def __aexit__(self, *args: Any) -> None:
            if asyncio.current_task() is not self._enter_task:
                raise RuntimeError(
                    "Attempted to exit cancel scope in a different task than it was entered in"
                )
            recorder.events.append(f"exit {self.name}")

        def _opened(self) -> Any:
            raise NotImplementedError

    class _Transport(_TaskBound):
        def __init__(self, name: str) -> None:
            self.name = name

        def _opened(self) -> tuple[str, str]:
            return ("read", "write")

    class ClientSession(_TaskBound):
        def __init__(self, read: str, write: str) -> None:
            counter["n"] += 1
            self.name = f"session{counter['n']}"

        def _opened(self) -> ClientSession:
            return self

        async def initialize(self) -> None:
            return None

        async def list_tools(self) -> Any:
            return types.SimpleNamespace(tools=[])

    class StdioServerParameters:
        def __init__(self, command: str, args: list[str], env: Any) -> None:
            self.command = command

    def stdio_client(params: StdioServerParameters) -> _Transport:
        return _Transport(f"stdio:{params.command}")

    def sse_client(url: str, headers: dict[str, str]) -> _Transport:
        return _Transport(f"sse:{url}")

    mcp = types.ModuleType("mcp")
    mcp.ClientSession = ClientSession  # type: ignore[attr-defined]
    mcp.StdioServerParameters = StdioServerParameters  # type: ignore[attr-defined]
    client_pkg = types.ModuleType("mcp.client")
    stdio_mod = types.ModuleType("mcp.client.stdio")
    stdio_mod.stdio_client = stdio_client  # type: ignore[attr-defined]
    sse_mod = types.ModuleType("mcp.client.sse")
    sse_mod.sse_client = sse_client  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", client_pkg)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.sse", sse_mod)


async def test_reconnecting_stdio_closes_the_previous_connection(monkeypatch):
    recorder = _Recorder()
    _install_fake_mcp(monkeypatch, recorder)
    client = MCPClient()

    await client.connect_stdio("server-a")
    await client.connect_stdio("server-b")

    assert recorder.events == [
        "enter stdio:server-a",
        "enter session1",
        "exit session1",
        "exit stdio:server-a",
        "enter stdio:server-b",
        "enter session2",
    ]


async def test_switching_from_stdio_to_sse_closes_the_stdio_connection(monkeypatch):
    recorder = _Recorder()
    _install_fake_mcp(monkeypatch, recorder)
    client = MCPClient()

    await client.connect_stdio("server-a")
    await client.connect_sse("http://localhost/sse")

    assert recorder.events == [
        "enter stdio:server-a",
        "enter session1",
        "exit session1",
        "exit stdio:server-a",
        "enter sse:http://localhost/sse",
        "enter session2",
    ]


async def test_close_after_reconnect_closes_only_the_live_connection(monkeypatch):
    recorder = _Recorder()
    _install_fake_mcp(monkeypatch, recorder)
    client = MCPClient()

    await client.connect_stdio("server-a")
    await client.connect_stdio("server-b")
    recorder.events.clear()

    await client.close()
    await client.close()

    assert recorder.events == ["exit session2", "exit stdio:server-b"]


async def test_close_from_a_different_task_still_tears_down(monkeypatch):
    # The transport/session are entered on MCPClient's own lifecycle task, so
    # closing from a *different* task than connect must still exit them (and
    # must not emit the best-effort RuntimeWarning). On the pre-#1099 code the
    # exit ran on the caller's task and the task-bound fakes would raise.
    recorder = _Recorder()
    _install_fake_mcp(monkeypatch, recorder)
    client = MCPClient()

    await asyncio.create_task(client.connect_stdio("server-a"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        await asyncio.create_task(client.close())

    assert recorder.events == [
        "enter stdio:server-a",
        "enter session1",
        "exit session1",
        "exit stdio:server-a",
    ]


async def test_reconnect_from_a_different_task_closes_the_previous_connection(monkeypatch):
    # Same as above, but the second connect (not close) runs on a new task and
    # must tear the first connection down cleanly before opening the new one.
    recorder = _Recorder()
    _install_fake_mcp(monkeypatch, recorder)
    client = MCPClient()

    await asyncio.create_task(client.connect_stdio("server-a"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        await asyncio.create_task(client.connect_stdio("server-b"))

    assert recorder.events == [
        "enter stdio:server-a",
        "enter session1",
        "exit session1",
        "exit stdio:server-a",
        "enter stdio:server-b",
        "enter session2",
    ]
