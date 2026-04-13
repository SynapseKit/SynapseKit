"""Core MCPServer implementation."""

from __future__ import annotations

import json
from typing import Any

from ...agents.base import BaseTool


class MCPServer:
    """Expose SynapseKit tools, RAG, or Agents as an MCP server.

    Requires: pip install mcp

    Usage::

        from synapsekit import MCPServer, CalculatorTool, DateTimeTool

        # Tools mode
        server = MCPServer(tools=[CalculatorTool(), DateTimeTool()])

        # RAG mode
        server = MCPServer(rag)

        # Agent mode
        server = MCPServer(agent)

        server.run()  # starts stdio server
    """

    #: Class names accepted as a source target (RAG / agent types).
    _SUPPORTED_TARGET_NAMES: frozenset[str] = frozenset(
        {"RAG", "RAGPipeline", "FunctionCallingAgent", "ReActAgent", "AgentExecutor"}
    )

    def __init__(
        self,
        target: Any = None,
        *,
        name: str | None = None,
        tools: list[BaseTool] | None = None,
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> None:
        # Backwards compatibility: if target is a string, it was passed as `name`
        if isinstance(target, str):
            name = target
            target = None

        # Validate target type — only None or known SynapseKit classes are allowed.
        if target is not None:
            cls_name = type(target).__name__
            if cls_name not in self._SUPPORTED_TARGET_NAMES:
                raise TypeError(
                    f"Unsupported source type: {type(target)!r}. "
                    f"MCPServer accepts RAG, FunctionCallingAgent, ReActAgent, "
                    f"or AgentExecutor as a positional argument. "
                    f"Pass individual tools via tools=[...]."
                )

        self._target = target
        self._name = name or "synapsekit"
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self._version = version
        self._server: Any = None

    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the server (tools mode)."""
        self._tools[tool.name] = tool

    def _build_server(self) -> Any:
        try:
            from mcp.server import Server
            from mcp.types import Resource, TextContent, Tool
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        server = Server(self._name)

        target = self._target

        # ------------------------------------------------------------------ #
        # RAG Mode
        # ------------------------------------------------------------------ #
        if target is not None and type(target).__name__ in ("RAG", "RAGPipeline"):
            @server.list_tools()
            async def list_tools() -> list[Tool]:
                return [
                    Tool(
                        name="rag_query",
                        description="Query the RAG knowledge base.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query"}
                            },
                            "required": ["query"]
                        }
                    )
                ]

            @server.call_tool()
            async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> list[TextContent]:
                if name == "rag_query":
                    query = (arguments or {}).get("query", "")
                    try:
                        # RAG has .ask() or .arun()
                        if hasattr(target, "ask"):
                            ans = await target.ask(query)
                        elif hasattr(target, "run"):
                            # some generic run
                            ans = await target.run(query)
                        else:
                            ans = "Target does not support ask() or run()"
                        return [TextContent(type="text", text=str(ans))]
                    except Exception as e:
                        return [TextContent(type="text", text=f"Error querying RAG: {e}")]
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            @server.list_resources()
            async def list_resources() -> list[Resource]:
                resources = []
                vs = getattr(target, "_vectorstore", None)
                if vs and hasattr(vs, "_texts") and hasattr(vs, "_metadata"):
                    for i, meta in enumerate(vs._metadata):
                        uri = f"document://{i}"
                        resources.append(
                            Resource(
                                uri=uri,
                                name=f"Document {i}",
                                mimeType="text/plain",
                                description=json.dumps(meta) if meta else "RAG Document"
                            )
                        )
                return resources

            @server.read_resource()
            async def read_resource(uri: str | Any) -> str | bytes:
                uri_str = str(uri)
                vs = getattr(target, "_vectorstore", None)
                if vs and hasattr(vs, "_texts") and uri_str.startswith("document://"):
                    try:
                        idx = int(uri_str.split("://")[1])
                        return vs._texts[idx]
                    except (ValueError, IndexError):
                        raise ValueError(f"Resource not found: {uri_str}")
                raise ValueError(f"Resource not found: {uri_str}")

        # ------------------------------------------------------------------ #
        # Agent Mode
        # ------------------------------------------------------------------ #
        elif target is not None and type(target).__name__ in ("FunctionCallingAgent", "AgentExecutor", "ReActAgent"):
            @server.list_tools()
            async def list_tools() -> list[Tool]:
                return [
                    Tool(
                        name="agent_run",
                        description="Send a task or query to the agent.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The task or question"}
                            },
                            "required": ["query"]
                        }
                    )
                ]

            @server.call_tool()
            async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> list[TextContent]:
                if name == "agent_run":
                    query = (arguments or {}).get("query", "")
                    try:
                        ans = await target.run(query)
                        return [TextContent(type="text", text=str(ans))]
                    except Exception as e:
                        return [TextContent(type="text", text=f"Error running agent: {e}")]
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        # ------------------------------------------------------------------ #
        # Tools Mode (default)
        # ------------------------------------------------------------------ #
        else:
            tools_map = self._tools

            @server.list_tools()
            async def list_tools() -> list[Tool]:
                result = []
                for t in tools_map.values():
                    result.append(
                        Tool(
                            name=t.name,
                            description=t.description,
                            inputSchema=t.parameters if hasattr(t, "parameters") else {},
                        )
                    )
                return result

            @server.call_tool()
            async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> list[TextContent]:
                tool = tools_map.get(name)
                if not tool:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

                try:
                    res = await tool.run(**(arguments or {}))
                    if res.is_error:
                        return [TextContent(type="text", text=f"Error: {res.error}")]
                    return [TextContent(type="text", text=res.output)]
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

    async def run_sse(self, host: str = "0.0.0.0", port: int = 8000, api_key: str | None = None) -> Any:
        """Run as SSE MCP server. Optionally handles API key auth if starlette is installed."""
        try:
            from mcp.server.sse import SseServerTransport
        except ImportError:
            raise ImportError("mcp package required: pip install mcp") from None

        server = self._build_server()
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Any) -> None:
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await server.run(streams[0], streams[1], server.create_initialization_options())

        self._sse_handler = handle_sse

        try:
            import uvicorn
            from starlette.applications import Starlette
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.responses import JSONResponse
            from starlette.routing import Route
        except ImportError:
            if api_key:
                raise ImportError("starlette and uvicorn are required for API key auth: pip install starlette uvicorn")
            return sse

        class AuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Any, call_next: Any) -> Any:
                if api_key:
                    auth = request.headers.get("Authorization")
                    if not auth or auth != f"Bearer {api_key}":
                        return JSONResponse({"error": "Unauthorized"}, status_code=401)
                return await call_next(request)

        async def messages(request: Any) -> None:
            await sse.handle_post_message(request.scope, request.receive, request._send)

        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Route("/messages/", endpoint=messages, methods=["POST"]),
            ]
        )
        if api_key:
            app.add_middleware(AuthMiddleware)

        # We need to run it, but run_sse is an async function. Wait, uvicorn.Server.serve() is async.
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()

