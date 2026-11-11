"""Reconnecting an MCPClient must close the connection it replaces."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from synapsekit.mcp.client import MCPClient


class _Recorder:
    """Records enter/exit of every fake transport and session, in order."""

    def __init__(self) -> None:
        self.events: list[str] = []


def _install_fake_mcp(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    counter = {"n": 0}

    class _Transport:
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aenter__(self) -> tuple[str, str]:
            recorder.events.append(f"enter {self.name}")
            return ("read", "write")

        async def __aexit__(self, *args: Any) -> None:
            recorder.events.append(f"exit {self.name}")

    class ClientSession:
        def __init__(self, read: str, write: str) -> None:
            counter["n"] += 1
            self.name = f"session{counter['n']}"

        async def __aenter__(self) -> ClientSession:
            recorder.events.append(f"enter {self.name}")
            return self

        async def __aexit__(self, *args: Any) -> None:
            recorder.events.append(f"exit {self.name}")

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
