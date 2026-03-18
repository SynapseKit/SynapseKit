"""MCP Server -- expose SynapseKit tools as an MCP server."""

from __future__ import annotations

from typing import Any

from ..agents.base import BaseTool


class MCPServer:
    """Expose SynapseKit tools as an MCP server.

    Requires: pip install mcp

    Usage::

        from synapsekit import MCPServer, CalculatorTool, DateTimeTool

        server = MCPServer(
            name="synapsekit-tools",
            tools=[CalculatorTool(), DateTimeTool()],
        )
        server.run()  # starts stdio server
    """

    def __init__(
        self,
        name: str = "synapsekit",
        tools: list[BaseTool] | None = None,
        version: str = "1.0.0",
    ) -> None:
        self._name = name
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self._version = version
        self._server: Any = None

    def add_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def _build_server(self) -> Any:
        try:
            from mcp.server import Server
            from mcp.types import TextContent, Tool
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        server = Server(self._name)
        tools = self._tools

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            result = []
            for t in tools.values():
                result.append(
                    Tool(
                        name=t.name,
                        description=t.description,
                        inputSchema=t.parameters if hasattr(t, "parameters") else {},
                    )
                )
            return result

        @server.call_tool()
        async def call_tool(
            name: str, arguments: dict[str, Any] | None = None
        ) -> list[TextContent]:
            tool = tools.get(name)
            if not tool:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            try:
                result = await tool.run(**(arguments or {}))
                if result.is_error:
                    return [TextContent(type="text", text=f"Error: {result.error}")]
                return [TextContent(type="text", text=result.output)]
            except Exception as e:
                return [TextContent(type="text", text=f"Tool error: {e}")]

        self._server = server
        return server

    def run(self) -> None:
        """Run as stdio MCP server (blocking)."""
        try:
            from mcp.server.stdio import stdio_server
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        import asyncio

        server = self._build_server()

        async def _run() -> None:
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        asyncio.run(_run())

    async def run_sse(self, host: str = "0.0.0.0", port: int = 8000) -> Any:
        """Run as SSE MCP server."""
        try:
            from mcp.server.sse import SseServerTransport
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        server = self._build_server()
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Any) -> None:
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())

        # Store the handler for integration with ASGI frameworks
        self._sse_handler = handle_sse
        return sse
