"""MCP Client -- connect SynapseKit agents to MCP tool servers."""

from __future__ import annotations

import asyncio
import copy
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..agents.base import BaseTool, ToolResult
from .pinning import (
    ToolApprovalRequiredError,
    ToolDefinition,
    ToolDriftDetectedError,
    ToolDriftWarning,
    ToolPinStore,
    make_tool_definition,
)


def _field(obj: Any, name: str, v2_name: str, default: Any = None) -> Any:
    # mcp 2.x renamed these attributes to snake_case.
    value = getattr(obj, name, None)
    return getattr(obj, v2_name, default) if value is None else value


class MCPToolAdapter(BaseTool):
    """Wraps an MCP tool as a SynapseKit BaseTool."""

    def __init__(self, mcp_tool: Any) -> None:
        self._mcp_tool = mcp_tool
        self.name = mcp_tool.name
        self.description = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
        self.parameters = _field(mcp_tool, "inputSchema", "input_schema") or {}

    async def run(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output="", error="Tool not connected to an MCP session.")


class MCPClient:
    """Connect to an MCP server and use its tools as SynapseKit tools.

    Requires: pip install mcp

    Usage::

        client = MCPClient()
        tools = await client.connect_stdio("uvx", ["some-mcp-server"])
        # tools is a list of BaseTool instances

        # Use with an agent
        executor = AgentExecutor(AgentConfig(llm=llm, tools=tools))
    """

    def __init__(
        self,
        *,
        server_id: str | None = None,
        pin_store_path: str | Path | None = None,
        block_on_drift: bool = True,
    ) -> None:
        if pin_store_path is not None and (not isinstance(server_id, str) or not server_id.strip()):
            raise ValueError("server_id must be a non-empty stable string when pinning is enabled")
        if not isinstance(block_on_drift, bool):
            raise TypeError("block_on_drift must be a bool")

        self._session: Any = None
        self._transition_lock = asyncio.Lock()

        self._tools: list[MCPToolAdapter] = []
        self._server_id = server_id
        self._pin_store = ToolPinStore(pin_store_path) if pin_store_path is not None else None
        self._block_on_drift = block_on_drift
        self._pending_tool_sources: list[tuple[Any, ToolDefinition]] | None = None
        self._tool_generation = 0
        self._active_tool_generation: int | None = None

    async def connect_stdio(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> list[BaseTool]:
        """Connect to an MCP server via stdio transport."""
        async with self._transition_lock:
            return await self._connect_stdio_unlocked(command, args, env)

    async def _connect_stdio_unlocked(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> list[BaseTool]:
        """Connect to an MCP server via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        params = StdioServerParameters(command=command, args=args or [], env=env)

        self._stdio_transport = stdio_client(params)
        streams = await self._stdio_transport.__aenter__()
        read_stream, write_stream = streams

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

        return await self._load_tools_unlocked()

    async def connect_sse(self, url: str, headers: dict[str, str] | None = None) -> list[BaseTool]:
        """Connect to an MCP server via SSE transport."""
        async with self._transition_lock:
            return await self._connect_sse_unlocked(url, headers)

    async def _connect_sse_unlocked(
        self, url: str, headers: dict[str, str] | None = None
    ) -> list[BaseTool]:
        """Connect to an MCP server via SSE transport."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        self._sse_transport = sse_client(url, headers=headers or {})
        streams = await self._sse_transport.__aenter__()
        read_stream, write_stream = streams

        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

        return await self._load_tools_unlocked()

    async def _load_tools(self) -> list[BaseTool]:
        """Load tools from the connected MCP server."""
        async with self._transition_lock:
            return await self._load_tools_unlocked()

    async def _load_tools_unlocked(self) -> list[BaseTool]:
        """Load tools from the connected MCP server."""
        if not self._session:
            raise RuntimeError("Not connected to an MCP server.")

        self._pending_tool_sources = None
        if self._pin_store is not None:
            self._tool_generation += 1
            self._active_tool_generation = None

        result = await self._session.list_tools()
        self._tools = []
        registered_tools = list(result.tools)

        if self._pin_store is None:
            self._tools = self._make_adapters(
                [(mcp_tool, None) for mcp_tool in registered_tools], generation=None
            )
            return list(self._tools)

        assert self._server_id is not None
        current_tools = self._snapshot_tool_sources(registered_tools)

        comparison = await asyncio.to_thread(
            self._pin_store.compare,
            self._server_id,
            [definition for _, definition in current_tools],
        )

        if (comparison.changed_tools or comparison.removed_tool_names) and self._block_on_drift:
            pending = [*comparison.unpinned_tools, *comparison.changed_tools]
            self._pending_tool_sources = current_tools
            raise ToolDriftDetectedError(
                [self._review_copy(definition) for definition in comparison.changed_tools],
                removed_tool_names=comparison.removed_tool_names,
                pending_tools=[self._review_copy(definition) for definition in pending],
            )

        if comparison.changed_tools or comparison.removed_tool_names:
            self._warn_tool_drift(comparison.changed_tools, comparison.removed_tool_names)

        if comparison.unpinned_tools:
            pending = [*comparison.unpinned_tools, *comparison.changed_tools]
            self._pending_tool_sources = current_tools
            raise ToolApprovalRequiredError(
                [self._review_copy(definition) for definition in pending]
            )

        generation = self._tool_generation
        self._active_tool_generation = generation
        self._tools = self._make_adapters(current_tools, generation=generation)
        return list(self._tools)

    @staticmethod
    def _snapshot_tool_sources(
        registered_tools: list[Any],
    ) -> list[tuple[Any, ToolDefinition]]:
        current_tools: list[tuple[Any, ToolDefinition]] = []
        seen_names: set[str] = set()
        for mcp_tool in registered_tools:
            definition = make_tool_definition(
                mcp_tool.name,
                getattr(mcp_tool, "description", None),
                _field(mcp_tool, "inputSchema", "input_schema"),
            )
            if definition.name in seen_names:
                raise ValueError(f"MCP server registered duplicate tool name {definition.name!r}")
            seen_names.add(definition.name)
            current_tools.append((mcp_tool, definition))
        return current_tools

    @staticmethod
    def _warn_tool_drift(
        changed_tools: tuple[ToolDefinition, ...],
        removed_tool_names: tuple[str, ...],
    ) -> None:
        details = []
        if changed_tools:
            names = ", ".join(json.dumps(tool.name) for tool in changed_tools)
            details.append(f"changed={len(changed_tools)} [{names}]")
        if removed_tool_names:
            names = ", ".join(json.dumps(name) for name in removed_tool_names)
            details.append(f"removed={len(removed_tool_names)} [{names}]")
        warnings.warn(
            f"MCP tool drift detected: {'; '.join(details)}",
            ToolDriftWarning,
            stacklevel=3,
        )

    @staticmethod
    def _review_copy(definition: ToolDefinition) -> ToolDefinition:
        return make_tool_definition(
            definition.name, definition.description, definition.input_schema
        )

    def _make_adapters(
        self,
        registered_tools: Sequence[tuple[Any, ToolDefinition | None]],
        *,
        generation: int | None,
    ) -> list[MCPToolAdapter]:
        adapters: list[MCPToolAdapter] = []
        session = self._session

        for mcp_tool, definition in registered_tools:
            adapter = MCPToolAdapter(mcp_tool)
            if definition is not None:
                adapter.name = definition.name
                adapter.description = definition.description or f"MCP tool: {definition.name}"
                adapter.parameters = copy.deepcopy(definition.input_schema) or {}

            def _make_run(tool_name: str, _session: Any = session):
                async def _run(**kwargs: Any) -> ToolResult:
                    if self._pin_store is not None and self._active_tool_generation != generation:
                        return ToolResult(
                            output="",
                            error="MCP tool definition is not currently approved.",
                        )
                    try:
                        call_result = await _session.call_tool(tool_name, arguments=kwargs)
                        output_parts = []
                        for content in call_result.content:
                            if hasattr(content, "text"):
                                output_parts.append(content.text)
                        output = (
                            "\n".join(output_parts) if output_parts else str(call_result.content)
                        )
                        if _field(call_result, "isError", "is_error", False):
                            return ToolResult(output="", error=output)
                        return ToolResult(output=output)
                    except Exception as e:
                        return ToolResult(output="", error=f"MCP tool error: {e}")

                return _run

            adapter.run = _make_run(mcp_tool.name)  # type: ignore[assignment]
            adapters.append(adapter)
        return adapters

    def _ensure_approval_state_current(
        self,
        session: Any,
        generation: int,
        pending_sources: list[tuple[Any, ToolDefinition]],
    ) -> None:
        if (
            self._session is not session
            or self._tool_generation != generation
            or self._pending_tool_sources is not pending_sources
        ):
            raise RuntimeError(
                "MCP tool approval state changed during review; "
                "review the current definitions and try again"
            )

    async def approve_current_tools(self) -> list[BaseTool]:
        """Re-check, pin, and return the reviewed current tool definitions.

        Call this only after reviewing ``pending_tools`` from the preceding
        ``ToolApprovalRequiredError`` or ``ToolDriftDetectedError`` exception.
        If the server definitions changed since they were shown, this raises a
        fresh review error instead of pinning or returning adapters.
        """
        async with self._transition_lock:
            return await self._approve_current_tools_unlocked()

    async def _approve_current_tools_unlocked(self) -> list[BaseTool]:
        """Re-check and activate reviewed tools while holding the transition lock."""
        if self._pin_store is None or self._server_id is None:
            raise RuntimeError("MCP tool pinning is not enabled")
        session = self._session
        generation = self._tool_generation
        pending_sources = self._pending_tool_sources
        if pending_sources is None:
            raise RuntimeError("there are no pending MCP tool definitions to approve")
        if not session:
            raise RuntimeError("Not connected to an MCP server.")

        reviewed_fingerprints = {
            definition.name: definition.fingerprint for _, definition in pending_sources
        }
        result = await session.list_tools()
        self._ensure_approval_state_current(session, generation, pending_sources)
        current_tools = self._snapshot_tool_sources(list(result.tools))
        current_fingerprints = {
            definition.name: definition.fingerprint for _, definition in current_tools
        }

        if current_fingerprints != reviewed_fingerprints:
            self._ensure_approval_state_current(session, generation, pending_sources)
            self._pending_tool_sources = current_tools
            self._tools = []
            self._active_tool_generation = None

            comparison = await asyncio.to_thread(
                self._pin_store.compare,
                self._server_id,
                [definition for _, definition in current_tools],
            )
            if (comparison.changed_tools or comparison.removed_tool_names) and self._block_on_drift:
                pending = [*comparison.unpinned_tools, *comparison.changed_tools]
                raise ToolDriftDetectedError(
                    [self._review_copy(definition) for definition in comparison.changed_tools],
                    removed_tool_names=comparison.removed_tool_names,
                    pending_tools=[self._review_copy(definition) for definition in pending],
                )
            if comparison.changed_tools or comparison.removed_tool_names:
                self._warn_tool_drift(comparison.changed_tools, comparison.removed_tool_names)

            pending = [*comparison.unpinned_tools, *comparison.changed_tools]
            if not pending:
                pending = [definition for _, definition in current_tools]
            raise ToolApprovalRequiredError(
                [self._review_copy(definition) for definition in pending],
                message=(
                    "MCP tool definitions changed since the last review; "
                    "review the refreshed definitions and approve again"
                ),
            )

        self._ensure_approval_state_current(session, generation, pending_sources)
        await asyncio.to_thread(
            self._pin_store.pin,
            self._server_id,
            [definition for _, definition in current_tools],
        )
        adapters = self._make_adapters(current_tools, generation=generation)
        self._ensure_approval_state_current(session, generation, pending_sources)
        self._tools = adapters
        self._active_tool_generation = generation
        self._pending_tool_sources = None
        return list(adapters)

    async def close(self) -> None:
        """Close the MCP connection."""
        async with self._transition_lock:
            await self._close_unlocked()

    async def _close_unlocked(self) -> None:
        """Close the MCP connection."""
        if self._pin_store is not None:
            self._active_tool_generation = None
            self._pending_tool_sources = None
            self._tools = []
        if self._session:
            await self._session.__aexit__(None, None, None)
        if hasattr(self, "_stdio_transport"):
            await self._stdio_transport.__aexit__(None, None, None)
        if hasattr(self, "_sse_transport"):
            await self._sse_transport.__aexit__(None, None, None)

    async def __aenter__(self) -> MCPClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
