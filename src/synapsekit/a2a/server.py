"""A2A Server -- expose a SynapseKit agent as an A2A endpoint."""

from __future__ import annotations

import json
from typing import Any

from ..agents.executor import AgentExecutor
from .agent_card import AgentCard
from .types import A2ATask


class A2AServer:
    """Expose a SynapseKit agent as an A2A-compatible server.

    Usage::
        server = A2AServer(
            executor=my_executor,
            card=AgentCard(name="my-agent", description="Helpful assistant"),
        )
        server.run(port=8001)
    """

    def __init__(self, executor: AgentExecutor, card: AgentCard) -> None:
        self._executor = executor
        self._card = card
        self._tasks: dict[str, A2ATask] = {}

    async def handle_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming JSON-RPC request."""
        method = body.get("method", "")
        request_id = body.get("id", "")
        params = body.get("params", {})

        if method == "tasks/send":
            return await self._handle_send(request_id, params)
        elif method == "tasks/get":
            return self._handle_get(request_id, params)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

    async def _handle_send(self, request_id: str, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id", request_id)
        message = params.get("message", {})
        content = message.get("content", "")

        task = A2ATask(id=task_id, state="running")
        task.add_message("user", content)
        self._tasks[task_id] = task

        try:
            result = await self._executor.run(content)
            task.add_message("agent", result)
            task.state = "completed"
        except Exception as e:
            task.add_message("agent", f"Error: {e}")
            task.state = "failed"

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": task.to_dict(),
        }

    def _handle_get(self, request_id: str, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)

        if not task:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": f"Task not found: {task_id}",
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": task.to_dict(),
        }

    def run(self, host: str = "0.0.0.0", port: int = 8001) -> None:
        """Run as HTTP server (stdlib only)."""
        import asyncio
        import http.server

        server_instance = self
        card = self._card

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/.well-known/agent.json":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(card.to_dict()).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/a2a":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length).decode())

                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(server_instance.handle_request(body))
                    loop.close()

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        httpd = http.server.HTTPServer((host, port), Handler)
        print(f"A2A Server running at http://{host}:{port}")
        httpd.serve_forever()
