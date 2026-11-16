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


def _contains_exception(error: BaseException, expected: type[BaseException], message: str) -> bool:
    if isinstance(error, expected) and str(error) == message:
        return True
    return any(
        _contains_exception(child, expected, message)
        for child in getattr(error, "exceptions", ())
        if isinstance(child, BaseException)
    )


async def test_child_failure_unwinds_owner_and_reports_original_error():
    from contextlib import asynccontextmanager

    import anyio

    from synapsekit.mcp.client import _MCPConnection

    fail_child, exited, session_exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owner = None

    @asynccontextmanager
    async def transport():
        nonlocal owner
        owner = asyncio.current_task()
        try:
            async with anyio.create_task_group() as group:

                async def child():
                    await fail_child.wait()
                    raise ValueError("child failure")

                group.start_soon(child)
                yield None, None
        finally:
            assert asyncio.current_task() is owner
            exited.set()

    class Session:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            assert asyncio.current_task() is owner
            await anyio.sleep(0)
            session_exited.set()

        async def initialize(self):
            pass

    connection = await _MCPConnection.open(transport, Session)
    fail_child.set()
    try:
        await asyncio.wait_for(exited.wait(), 2)
        await asyncio.wait_for(asyncio.shield(connection._task), 2)
        assert session_exited.is_set()
        assert connection._done.done()
        with pytest.raises(BaseException) as error:
            await asyncio.wait_for(connection.aclose(), 2)
        assert _contains_exception(error.value, ValueError, "child failure")

        @asynccontextmanager
        async def healthy_transport():
            nonlocal owner
            owner = asyncio.current_task()
            yield None, None

        replacement = await asyncio.wait_for(_MCPConnection.open(healthy_transport, Session), 2)
        await asyncio.wait_for(replacement.aclose(), 2)
        assert replacement._task.done() and replacement._done.done()
    finally:
        if not connection._task.done():
            connection._task.cancel()
            await asyncio.gather(connection._task, return_exceptions=True)


async def test_child_failure_during_initialization_unwinds_and_reports_child_error():
    from contextlib import asynccontextmanager

    import anyio

    from synapsekit.mcp.client import _MCPConnection

    trigger = asyncio.Event()
    session_exited, transport_exited = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def transport():
        async with anyio.create_task_group() as group:

            async def fail():
                await trigger.wait()
                raise ValueError("initialize child failure")

            group.start_soon(fail)
            try:
                yield None, None
            finally:
                transport_exited.set()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await anyio.sleep(0)
            session_exited.set()

        async def initialize(self):
            trigger.set()
            await anyio.sleep(1)

    with pytest.raises(BaseException) as error:
        await asyncio.wait_for(_MCPConnection.open(transport, lambda *_: Session()), 2)
    assert _contains_exception(error.value, ValueError, "initialize child failure")
    assert session_exited.is_set()
    assert transport_exited.is_set()


async def test_abandoned_child_failure_does_not_log_unretrieved_future():
    import gc
    from contextlib import asynccontextmanager

    import anyio

    from synapsekit.mcp.client import _MCPConnection

    trigger, exited = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def transport():
        try:
            async with anyio.create_task_group() as group:

                async def fail():
                    await trigger.wait()
                    raise ValueError("abandoned child failure")

                group.start_soon(fail)
                yield None, None
        finally:
            exited.set()

    class Session:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

    loop = asyncio.get_running_loop()
    errors = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    connection = await _MCPConnection.open(transport, Session)
    task = connection._task
    try:
        trigger.set()
        await asyncio.wait_for(exited.wait(), 2)
        await asyncio.wait_for(asyncio.shield(task), 2)
        del connection
        gc.collect()
        await asyncio.sleep(0)
        assert not errors
    finally:
        loop.set_exception_handler(previous)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


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


@pytest.mark.parametrize("repeat_cancel", [False, True], ids=["single", "repeated"])
@pytest.mark.parametrize("transport", ["stdio", "sse"])
async def test_cancel_during_initialization_cleans_up_and_allows_reconnect(
    monkeypatch, transport, repeat_cancel
):
    recorder = _Recorder()
    _install_fake_mcp(monkeypatch, recorder)
    session_type = sys.modules["mcp"].ClientSession
    initialize_started, release_initialize = asyncio.Event(), asyncio.Event()
    owner = None
    original_initialize = session_type.initialize

    async def blocked_initialize(self):
        nonlocal owner
        if owner is None:
            owner = asyncio.current_task()
            initialize_started.set()
            await release_initialize.wait()
        await original_initialize(self)

    monkeypatch.setattr(session_type, "initialize", blocked_initialize)
    client = MCPClient()

    async def connect():
        if transport == "stdio":
            return await client.connect_stdio("server-a")
        return await client.connect_sse("http://localhost/sse")

    task = asyncio.create_task(connect())
    try:
        await asyncio.wait_for(initialize_started.wait(), 5)
        owner_done = owner.get_coro().cr_frame.f_locals["done"]
        task.cancel("first cancellation")
        await asyncio.sleep(0)
        if repeat_cancel:
            task.cancel("second cancellation")
        release_initialize.set()
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await asyncio.wait_for(task, 5)
        await asyncio.wait_for(asyncio.shield(owner), 5)
        assert owner.done() and owner.exception() is None
        assert owner_done.done() and not owner_done.cancelled() and owner_done.exception() is None
        assert "exit session1" in recorder.events
        assert any(
            event.startswith("exit stdio:") or event.startswith("exit sse:")
            for event in recorder.events
        )
        assert client._connection is None and client._session is None
        await asyncio.wait_for(connect(), 5)
        replacement = client._connection
        await asyncio.wait_for(client.close(), 5)
        assert replacement._task.done() and replacement._done.done()
    finally:
        release_initialize.set()
        if owner is not None and not owner.done():
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_cancel_while_failed_initialization_is_unwinding_does_not_interrupt_exit(monkeypatch):
    recorder = _Recorder()
    _install_fake_mcp(monkeypatch, recorder)
    session_type = sys.modules["mcp"].ClientSession
    exit_started, release_exit, exit_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owner = None
    original_exit = session_type.__aexit__

    async def fail_initialize(self):
        raise RuntimeError("initialize failed")

    async def blocked_exit(self, *args):
        nonlocal owner
        owner = asyncio.current_task()
        exit_started.set()
        await release_exit.wait()
        await original_exit(self, *args)
        exit_finished.set()

    monkeypatch.setattr(session_type, "initialize", fail_initialize)
    monkeypatch.setattr(session_type, "__aexit__", blocked_exit)
    client = MCPClient()
    task = asyncio.create_task(client.connect_stdio("server-a"))
    try:
        await asyncio.wait_for(exit_started.wait(), 5)
        task.cancel("caller cancelled")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release_exit.set()
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await asyncio.wait_for(task, 5)
        await asyncio.wait_for(asyncio.shield(owner), 5)
        assert exit_finished.is_set()
        assert "exit session1" in recorder.events
        assert "exit stdio:server-a" in recorder.events
    finally:
        release_exit.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if owner is not None and not owner.done():
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.parametrize("transport", ["stdio", "sse"])
async def test_cancel_during_transport_exit_cleans_up_and_allows_reconnect(monkeypatch, transport):
    recorder = _Recorder()
    _install_fake_mcp(monkeypatch, recorder)
    module = sys.modules[f"mcp.client.{transport}"]
    factory_name = f"{transport}_client"
    original_factory = getattr(module, factory_name)
    exit_started, release_exit = asyncio.Event(), asyncio.Event()
    exit_completed = asyncio.Event()

    def factory(*args, **kwargs):
        context = original_factory(*args, **kwargs)
        original_exit = type(context).__aexit__

        class BlockingExit(type(context)):
            async def __aexit__(self, *args):
                if not exit_started.is_set():
                    exit_started.set()
                    await release_exit.wait()
                result = await original_exit(self, *args)
                exit_completed.set()
                return result

        context.__class__ = BlockingExit
        return context

    monkeypatch.setattr(module, factory_name, factory)
    client = MCPClient()

    async def connect():
        if transport == "stdio":
            return await client.connect_stdio("server-a")
        return await client.connect_sse("http://localhost/sse")

    await connect()
    connection = client._connection
    task = asyncio.create_task(client.close())
    try:
        await asyncio.wait_for(exit_started.wait(), 5)
        task.cancel("first cancellation")
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await asyncio.wait_for(task, 5)
        another_close = asyncio.create_task(connection.aclose())
        await asyncio.sleep(0)
        another_close.cancel("second cancellation")
        with pytest.raises(asyncio.CancelledError, match="second cancellation"):
            await asyncio.wait_for(another_close, 5)
        release_exit.set()
        await asyncio.wait_for(asyncio.shield(connection._task), 5)
        assert exit_completed.is_set()
        assert connection._done.done() and not connection._done.cancelled()
        assert connection._done.exception() is None
        await asyncio.wait_for(connect(), 5)
        replacement = client._connection
        await asyncio.wait_for(client.close(), 5)
        assert replacement._task.done() and replacement._done.done()
    finally:
        release_exit.set()
        if not connection._task.done():
            connection._task.cancel()
            await asyncio.gather(connection._task, return_exceptions=True)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


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
