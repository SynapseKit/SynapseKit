"""A2A Client -- send tasks to remote agents."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

from .types import A2ATask


class A2AClient:
    """Client for communicating with A2A-compatible agents.

    Usage::
        client = A2AClient(endpoint="http://localhost:8001")
        card = await client.get_agent_card()
        task = await client.send_task("Research AI trends in 2026")
    """

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint.rstrip("/")

    async def get_agent_card(self) -> dict[str, Any]:
        """Fetch the agent's capability card."""
        return await self._get("/.well-known/agent.json")

    async def send_task(self, message: str, task_id: str | None = None) -> A2ATask:
        """Send a task to the remote agent."""
        import uuid

        tid = task_id or str(uuid.uuid4())

        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "id": tid,
            "params": {
                "id": tid,
                "message": {"role": "user", "content": message},
            },
        }

        result = await self._post("/a2a", payload)

        task = A2ATask(id=tid)
        task.add_message("user", message)

        if isinstance(result, dict):
            task.state = result.get("result", {}).get("state", "completed")
            agent_messages = result.get("result", {}).get("messages", [])
            for msg in agent_messages:
                if msg.get("role") == "agent":
                    task.add_message("agent", msg.get("content", ""))
            task.artifacts = result.get("result", {}).get("artifacts", [])

        return task

    async def _get(self, path: str) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        url = f"{self._endpoint}{path}"

        def _fetch():
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())

        return await loop.run_in_executor(None, _fetch)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        url = f"{self._endpoint}{path}"
        data = json.dumps(payload).encode()

        def _fetch():
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())

        return await loop.run_in_executor(None, _fetch)
