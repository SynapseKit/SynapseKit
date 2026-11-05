from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import inspect
import json
import sys
import threading
import types
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from synapsekit.mcp.client import MCPClient


def _pinning_api(*names: str) -> Any:
    module = importlib.import_module("synapsekit.mcp.pinning")
    missing = [name for name in names if not hasattr(module, name)]
    assert not missing, f"missing MCP pinning API: {', '.join(missing)}"
    return module


def _fingerprint(module: Any, description: str, schema: dict[str, Any]) -> str:
    return module.tool_fingerprint("lookup", description, schema)


def test_tool_fingerprint_is_independent_of_schema_key_order():
    module = _pinning_api("tool_fingerprint")
    first = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
    }
    reordered = {
        "properties": {"limit": {"type": "integer"}, "query": {"type": "string"}},
        "type": "object",
    }

    assert _fingerprint(module, "Search records", first) == _fingerprint(
        module, "Search records", reordered
    )


def test_tool_fingerprint_changes_when_description_or_schema_changes():
    module = _pinning_api("tool_fingerprint")
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    baseline = _fingerprint(module, "Search records", schema)

    assert _fingerprint(module, "Search private records", schema) != baseline
    assert (
        _fingerprint(
            module,
            "Search records",
            {"type": "object", "properties": {"query": {"type": "integer"}}},
        )
        != baseline
    )


def test_pin_store_survives_reload_and_persists_only_fingerprints(tmp_path):
    module = _pinning_api("ToolPinStore")
    path = tmp_path / "tool-pins.json"
    fingerprint = "a" * 64
    store = module.ToolPinStore(path)
    store.set_pins("catalog-prod", {"lookup": fingerprint})

    assert module.ToolPinStore(path).get_pins("catalog-prod") == {"lookup": fingerprint}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"version": 1, "servers": {"catalog-prod": {"lookup": fingerprint}}}
    assert "description" not in path.read_text(encoding="utf-8")


def test_separate_pin_store_instances_do_not_lose_distinct_server_updates(tmp_path, monkeypatch):
    module = _pinning_api("ToolPinStore")
    path = tmp_path / "tool-pins.json"
    first_read_started = threading.Event()
    release_first_read = threading.Event()

    class _TrackingLock:
        def __init__(self):
            self._lock = threading.Lock()
            self._attempts_lock = threading.Lock()
            self._attempt_count = 0
            self.second_acquire_attempted = threading.Event()

        def __enter__(self):
            with self._attempts_lock:
                self._attempt_count += 1
                if self._attempt_count == 2:
                    self.second_acquire_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()

    class _BlockingReadToolPinStore(module.ToolPinStore):
        def _read_all(self):
            payload = super()._read_all()
            first_read_started.set()
            release_first_read.wait()
            return payload

    tracking_lock = _TrackingLock()
    monkeypatch.setattr(module, "_STORE_WRITE_LOCK", tracking_lock)
    stores = [_BlockingReadToolPinStore(path), module.ToolPinStore(path)]
    fingerprints = ["a" * 64, "b" * 64]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(stores[0].set_pins, "server-a", {"lookup": fingerprints[0]})
        second_future = None
        second_lock_attempted = False
        try:
            assert first_read_started.wait(timeout=10), "first writer did not enter _read_all"
            second_future = pool.submit(stores[1].set_pins, "server-b", {"lookup": fingerprints[1]})
            second_lock_attempted = tracking_lock.second_acquire_attempted.wait(timeout=10)
        finally:
            release_first_read.set()

        first_future.result()
        if second_future is not None:
            second_future.result()

    assert second_lock_attempted, "second writer never attempted the shared write lock"
    reloaded = module.ToolPinStore(path)
    assert reloaded.get_pins("server-a") == {"lookup": fingerprints[0]}
    assert reloaded.get_pins("server-b") == {"lookup": fingerprints[1]}


def test_pin_store_pin_and_compare_report_unpinned_changed_and_removed_tools(tmp_path):
    module = _pinning_api("ToolPinStore", "make_tool_definition")
    store = module.ToolPinStore(tmp_path / "tool-pins.json")
    assert callable(getattr(store, "pin", None))
    assert callable(getattr(store, "compare", None))

    lookup = module.make_tool_definition("lookup", "Public lookup", {"type": "object"})
    search_before = module.make_tool_definition("search", "Old search", {"type": "object"})
    retired = module.make_tool_definition("retired", "Retired tool", {"type": "object"})
    store.pin("catalog-prod", [lookup, search_before, retired])

    search_after = module.make_tool_definition("search", "Updated search", {"type": "object"})
    newcomer = module.make_tool_definition("newcomer", "New tool", {"type": "object"})
    comparison = store.compare("catalog-prod", [lookup, search_after, newcomer])

    assert tuple(tool.name for tool in comparison.unpinned_tools) == ("newcomer",)
    assert tuple(tool.name for tool in comparison.changed_tools) == ("search",)
    assert comparison.removed_tool_names == ("retired",)
    with pytest.raises(FrozenInstanceError):
        comparison.removed_tool_names = ()

    saved = module.ToolPinStore(store.path).get_pins("catalog-prod")
    assert saved == {
        "lookup": lookup.fingerprint,
        "search": search_before.fingerprint,
        "retired": retired.fingerprint,
    }
    payload = store.path.read_text(encoding="utf-8")
    assert "Old search" not in payload
    assert "input_schema" not in payload


@pytest.mark.parametrize(
    "contents",
    [
        "{not-json",
        '{"version": 2, "servers": {}}',
    ],
    ids=["corrupt-json", "unsupported-version"],
)
def test_pin_store_rejects_corrupt_or_unsupported_data(tmp_path, contents):
    module = _pinning_api("ToolPinStore", "ToolPinStoreError")
    path = tmp_path / "tool-pins.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(module.ToolPinStoreError):
        module.ToolPinStore(path).get_pins("catalog-prod")


def test_pinning_requires_an_explicit_stable_server_id(tmp_path):
    _pinning_api()

    with pytest.raises(ValueError, match="server_id"):
        MCPClient(pin_store_path=tmp_path / "tool-pins.json")


class _Tool:
    def __init__(self, description: str, schema: dict[str, Any] | None = None) -> None:
        self.name = "lookup"
        self.description = description
        self.inputSchema = schema or {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }


class _Session:
    def __init__(self, *tools: _Tool) -> None:
        self.tools = list(tools)

    async def list_tools(self) -> Any:
        return type("ListToolsResult", (), {"tools": self.tools})()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        content = [type("Text", (), {"text": f"{name}:{arguments['query']}"})()]
        return type("CallResult", (), {"content": content, "isError": False})()


def _client(path, tool: _Tool, **kwargs: Any) -> MCPClient:
    client = MCPClient(server_id="catalog-prod", pin_store_path=path, **kwargs)
    client._session = _Session(tool)
    return client


@pytest.mark.parametrize("block_on_drift", [True, False])
async def test_unpinned_tools_require_explicit_approval(tmp_path, block_on_drift):
    module = _pinning_api("ToolPinStore", "ToolApprovalRequiredError")
    parameters = inspect.signature(MCPClient).parameters
    assert {"server_id", "pin_store_path"} <= set(parameters)
    assert hasattr(MCPClient, "approve_current_tools")

    path = tmp_path / "tool-pins.json"
    client = _client(path, _Tool("Search records"), block_on_drift=block_on_drift)
    with pytest.raises(module.ToolApprovalRequiredError) as error:
        await client._load_tools()

    assert [definition.name for definition in error.value.pending_tools] == ["lookup"]
    assert error.value.pending_tools[0].description == "Search records"
    assert client._tools == []

    adapters = await client.approve_current_tools()
    assert [tool.name for tool in adapters] == ["lookup"]
    assert module.ToolPinStore(path).get_pins("catalog-prod")


async def test_approval_requires_rereview_when_tools_change_after_review(tmp_path):
    module = _pinning_api("ToolPinStore", "ToolApprovalRequiredError")
    path = tmp_path / "tool-pins.json"
    tool = _Tool("Search records")
    client = _client(path, tool)

    with pytest.raises(module.ToolApprovalRequiredError):
        await client._load_tools()

    tool.description = "Changed after review"
    with pytest.raises(module.ToolApprovalRequiredError) as refreshed_error:
        await client.approve_current_tools()

    assert client._tools == []
    assert client._active_tool_generation is None
    assert module.ToolPinStore(path).get_pins("catalog-prod") == {}
    assert "changed since the last review" in str(refreshed_error.value)
    assert [item.description for item in refreshed_error.value.pending_tools] == [
        "Changed after review"
    ]

    (adapter,) = await client.approve_current_tools()

    assert adapter.description == "Changed after review"
    assert module.ToolPinStore(path).get_pins("catalog-prod") == {
        "lookup": module.tool_fingerprint("lookup", "Changed after review", tool.inputSchema)
    }


async def test_unchanged_tools_load_after_their_fingerprints_are_pinned(tmp_path):
    module = _pinning_api("ToolPinStore", "tool_fingerprint")
    path = tmp_path / "tool-pins.json"
    tool = _Tool("Search records")
    fingerprint = module.tool_fingerprint(tool.name, tool.description, tool.inputSchema)
    module.ToolPinStore(path).set_pins("catalog-prod", {tool.name: fingerprint})
    client = _client(path, tool)

    (adapter,) = await client._load_tools()

    assert adapter.name == "lookup"
    result = await adapter.run(query="alpha")
    assert result.output == "lookup:alpha"


async def test_drift_blocks_adapters_before_any_tool_is_returned(tmp_path):
    module = _pinning_api("ToolPinStore", "tool_fingerprint", "ToolDriftDetectedError")
    path = tmp_path / "tool-pins.json"
    original = _Tool("Search records")
    original_fingerprint = module.tool_fingerprint(
        original.name, original.description, original.inputSchema
    )
    module.ToolPinStore(path).set_pins("catalog-prod", {original.name: original_fingerprint})
    changed = _Tool(
        "Search records",
        {"type": "object", "properties": {"query": {"type": "integer"}}},
    )
    client = _client(path, changed)

    with pytest.raises(module.ToolDriftDetectedError):
        await client._load_tools()

    assert client._tools == []


async def test_report_only_drift_warns_with_names_and_returns_adapters(tmp_path):
    module = _pinning_api("ToolPinStore", "tool_fingerprint", "ToolDriftWarning")
    path = tmp_path / "tool-pins.json"
    original = _Tool("Original description")
    removed_fingerprint = "b" * 64
    module.ToolPinStore(path).set_pins(
        "catalog-prod",
        {
            original.name: module.tool_fingerprint(
                original.name, original.description, original.inputSchema
            ),
            "retired": removed_fingerprint,
        },
    )
    changed = _Tool("Sensitive changed description")
    client = _client(path, changed, block_on_drift=False)

    with pytest.warns(module.ToolDriftWarning) as caught:
        adapters = await client._load_tools()

    assert [tool.name for tool in adapters] == ["lookup"]
    warning_text = str(caught[0].message)
    assert warning_text == ('MCP tool drift detected: changed=1 ["lookup"]; removed=1 ["retired"]')
    assert "Sensitive changed description" not in warning_text
    assert "query" not in warning_text


async def test_report_only_new_tools_require_review_of_changed_definitions(tmp_path):
    module = _pinning_api(
        "ToolPinStore", "tool_fingerprint", "ToolDriftWarning", "ToolApprovalRequiredError"
    )
    path = tmp_path / "tool-pins.json"
    original = _Tool("Original lookup description")
    retired_fingerprint = "b" * 64
    module.ToolPinStore(path).set_pins(
        "catalog-prod",
        {
            original.name: module.tool_fingerprint(
                original.name, original.description, original.inputSchema
            ),
            "retired": retired_fingerprint,
        },
    )

    changed_schema = {"type": "object", "properties": {"query": {"type": "integer"}}}
    changed = _Tool("Current lookup description", changed_schema)
    new_schema = {"type": "object", "properties": {"limit": {"type": "integer"}}}
    new_tool = _Tool("Current new-tool description", new_schema)
    new_tool.name = "new-tool"
    client = MCPClient(server_id="catalog-prod", pin_store_path=path, block_on_drift=False)
    client._session = _Session(changed, new_tool)

    with pytest.warns(module.ToolDriftWarning) as caught:
        with pytest.raises(module.ToolApprovalRequiredError) as error:
            await client._load_tools()

    pending = {definition.name: definition for definition in error.value.pending_tools}
    assert set(pending) == {"lookup", "new-tool"}
    assert pending["lookup"].description == "Current lookup description"
    assert pending["lookup"].input_schema == changed_schema
    assert pending["new-tool"].description == "Current new-tool description"
    assert pending["new-tool"].input_schema == new_schema
    assert 'removed=1 ["retired"]' in str(caught[0].message)
    assert client._tools == []


async def test_report_only_drift_json_escapes_control_characters_in_tool_names(tmp_path):
    module = _pinning_api("ToolPinStore", "tool_fingerprint", "ToolDriftWarning")
    path = tmp_path / "tool-pins.json"
    unsafe_name = "lookup\n\x1b[31mforged"
    original = _Tool("Original description")
    original.name = unsafe_name
    module.ToolPinStore(path).set_pins(
        "catalog-prod",
        {
            unsafe_name: module.tool_fingerprint(
                unsafe_name, original.description, original.inputSchema
            )
        },
    )
    changed = _Tool("Changed description")
    changed.name = unsafe_name
    client = _client(path, changed, block_on_drift=False)

    with pytest.warns(module.ToolDriftWarning) as caught:
        await client._load_tools()

    warning_text = str(caught[0].message)
    assert '"lookup\\n\\u001b[31mforged"' in warning_text
    assert not any(ord(character) < 0x20 for character in warning_text)


async def test_changed_description_is_detected_on_reconnect(tmp_path):
    module = _pinning_api("ToolPinStore", "tool_fingerprint", "ToolDriftDetectedError")
    path = tmp_path / "tool-pins.json"
    original = _Tool("Search public records")
    original_fingerprint = module.tool_fingerprint(
        original.name, original.description, original.inputSchema
    )
    module.ToolPinStore(path).set_pins("catalog-prod", {original.name: original_fingerprint})
    client = _client(path, _Tool("Search private records"))

    with pytest.raises(module.ToolDriftDetectedError) as error:
        await client._load_tools()

    assert [definition.name for definition in error.value.changed_tools] == ["lookup"]
    assert error.value.changed_tools[0].description == "Search private records"


async def test_client_without_pin_store_preserves_existing_behavior():
    client = MCPClient()
    client._session = _Session(_Tool("Search records"))

    (tool,) = await client._load_tools()

    assert tool.description == "Search records"
    assert (await tool.run(query="alpha")).output == "lookup:alpha"


async def test_duplicate_tool_names_are_rejected_before_adapters_are_returned(tmp_path):
    _pinning_api("ToolPinStore")
    client = MCPClient(server_id="catalog-prod", pin_store_path=tmp_path / "tool-pins.json")
    client._session = _Session(_Tool("First definition"), _Tool("Second definition"))

    with pytest.raises(ValueError, match="duplicate tool name 'lookup'"):
        await client._load_tools()

    assert client._tools == []


async def test_concurrent_reconnect_waits_for_approval_before_loading_drift(tmp_path, monkeypatch):
    module = _pinning_api("ToolApprovalRequiredError", "ToolDriftDetectedError", "ToolPinStore")
    approval_list_started = asyncio.Event()
    release_approval_list = asyncio.Event()
    b_transport_started = asyncio.Event()
    b_initialize_started = asyncio.Event()
    release_b_initialize = asyncio.Event()

    class _InterleavingSession:
        def __init__(self, tag: str, tool: _Tool) -> None:
            self.tag = tag
            self.tools = [tool]
            self.list_calls = 0

        async def __aenter__(self) -> _InterleavingSession:
            return self

        async def initialize(self) -> None:
            if self.tag == "B":
                b_initialize_started.set()
                await release_b_initialize.wait()

        async def list_tools(self) -> Any:
            self.list_calls += 1
            if self.tag == "A" and self.list_calls == 2:
                approval_list_started.set()
                await release_approval_list.wait()
            return types.SimpleNamespace(tools=self.tools)

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            content = [types.SimpleNamespace(text=f"{self.tag}:{arguments['query']}")]
            return types.SimpleNamespace(content=content, isError=False)

        async def __aexit__(self, *args: Any) -> None:
            return None

    class _Transport:
        def __init__(self, session: _InterleavingSession) -> None:
            self.session = session

        async def __aenter__(self) -> tuple[_InterleavingSession, object]:
            return self.session, object()

        async def __aexit__(self, *args: Any) -> None:
            return None

    session_a = _InterleavingSession("A", _Tool("Reviewed definition"))
    session_b = _InterleavingSession("B", _Tool("Changed definition"))

    class _StdioServerParameters:
        def __init__(self, command: str, args: list[str], env: dict[str, str] | None) -> None:
            self.command = command
            self.args = args
            self.env = env

    mcp_module = types.ModuleType("mcp")
    mcp_client_module = types.ModuleType("mcp.client")
    stdio_module = types.ModuleType("mcp.client.stdio")
    mcp_module.ClientSession = lambda read_stream, write_stream: read_stream
    mcp_module.StdioServerParameters = _StdioServerParameters

    def _stdio_client(params: _StdioServerParameters) -> _Transport:
        if params.command == "server-a":
            return _Transport(session_a)
        assert params.command == "server-b"
        b_transport_started.set()
        return _Transport(session_b)

    stdio_module.stdio_client = _stdio_client
    mcp_module.client = mcp_client_module
    mcp_client_module.stdio = stdio_module
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", mcp_client_module)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_module)

    client = MCPClient(server_id="catalog-prod", pin_store_path=tmp_path / "tool-pins.json")
    with pytest.raises(module.ToolApprovalRequiredError):
        await client.connect_stdio("server-a")

    approval_task = asyncio.create_task(client.approve_current_tools())
    await approval_list_started.wait()

    b_connect_started = asyncio.Event()

    async def _connect_b() -> list[Any]:
        b_connect_started.set()
        return await client.connect_stdio("server-b")

    b_task = asyncio.create_task(_connect_b())
    await b_connect_started.wait()

    try:
        assert not b_transport_started.is_set()
        release_approval_list.set()
        (approved_adapter,) = await approval_task
        await b_initialize_started.wait()

        approved_result = await approved_adapter.run(query="alpha")
        assert approved_result.output == "A:alpha"

        release_b_initialize.set()
        with pytest.raises(module.ToolDriftDetectedError) as drift_error:
            await b_task

        assert [tool.description for tool in drift_error.value.changed_tools] == [
            "Changed definition"
        ]
        assert module.ToolPinStore(client._pin_store.path).get_pins("catalog-prod") == {
            "lookup": module.make_tool_definition(
                "lookup", "Reviewed definition", session_a.tools[0].inputSchema
            ).fingerprint
        }
        assert client._tools == []
        assert client._active_tool_generation is None
        assert [definition.description for _, definition in client._pending_tool_sources] == [
            "Changed definition"
        ]
    finally:
        release_approval_list.set()
        release_b_initialize.set()
        await asyncio.gather(approval_task, b_task, return_exceptions=True)
