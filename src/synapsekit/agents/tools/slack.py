"""Slack Tool: send messages via Slack webhook or Web API bot token."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class SlackTool(BaseTool):
    """Send messages to Slack channels via webhook URL or Bot API token.

    Supports two modes:

    * **Webhook** — post to an incoming-webhook URL (no scopes needed).
    * **Bot token** — use ``chat.postMessage`` via the Web API.

    Config via constructor args or env vars ``SLACK_WEBHOOK_URL`` /
    ``SLACK_BOT_TOKEN``.  Stdlib ``urllib`` only — no extra dependencies.

    Usage::

        tool = SlackTool(webhook_url="https://hooks.slack.com/services/...")
        result = await tool.run(action="send_webhook", text="Hello from SynapseKit!")

        tool = SlackTool(bot_token="xoxb-...")
        result = await tool.run(action="send_message", channel="#general", text="Hi!")
    """

    name = "slack"
    description = (
        "Send messages to Slack via webhook or bot token. "
        "Actions: send_message (bot token), send_webhook (webhook URL)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["send_message", "send_webhook"],
            },
            "channel": {
                "type": "string",
                "description": "Slack channel for send_message (e.g. '#general')",
            },
            "text": {
                "type": "string",
                "description": "Message text to send",
            },
        },
        "required": ["action", "text"],
    }

    def __init__(
        self,
        webhook_url: str | None = None,
        bot_token: str | None = None,
    ) -> None:
        self._webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self._bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN")

    async def run(self, action: str = "", text: str = "", **kwargs: Any) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified.")
        if not text:
            return ToolResult(output="", error="No text provided.")

        handlers = {
            "send_message": self._send_message,
            "send_webhook": self._send_webhook,
        }

        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                output="",
                error=f"Unknown action: {action}. Must be one of: {', '.join(handlers)}",
            )

        try:
            return await handler(text=text, **kwargs)
        except Exception as e:
            return ToolResult(output="", error=f"Slack error: {e}")

    async def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> str:
        import asyncio
        import json
        import urllib.request

        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        loop = asyncio.get_event_loop()

        def _fetch():
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode()

        return await loop.run_in_executor(None, _fetch)

    async def _send_webhook(self, text: str = "", **kwargs: Any) -> ToolResult:
        if not self._webhook_url:
            return ToolResult(output="", error="No SLACK_WEBHOOK_URL configured.")

        headers = {"Content-Type": "application/json"}
        await self._post_json(self._webhook_url, {"text": text}, headers)
        return ToolResult(output="Message sent via webhook.")

    async def _send_message(self, text: str = "", channel: str = "", **kwargs: Any) -> ToolResult:
        if not self._bot_token:
            return ToolResult(output="", error="No SLACK_BOT_TOKEN configured.")
        if not channel:
            return ToolResult(output="", error="No channel provided for send_message.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._bot_token}",
        }
        body = await self._post_json(
            "https://slack.com/api/chat.postMessage",
            {"channel": channel, "text": text},
            headers,
        )

        import json

        resp = json.loads(body)
        if not resp.get("ok"):
            return ToolResult(output="", error=f"Slack API error: {resp.get('error', 'unknown')}")
        return ToolResult(output=f"Message sent to {channel}.")
