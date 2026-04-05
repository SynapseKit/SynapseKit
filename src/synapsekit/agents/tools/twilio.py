"""Twilio Tool: send SMS and WhatsApp messages via Twilio Messaging API.

WARNING: this tool can send messages to arbitrary phone numbers.
A security warning is logged on instantiation. Use with caution in
agent pipelines — consider rate limiting at the agent level.

WhatsApp caveats:
  - Sandbox only works with numbers that opted in (user sends join code first)
  - Sandbox number can change periodically, breaking hardcoded integrations
  - 24-hour messaging window: freeform messages only within 24h of last user
    message; outside that window you need pre-approved message templates
  - Production requires a Twilio WhatsApp sender (Meta Business verification,
    takes 1-2 weeks)
  - Rate limits are tier-based (1K/10K/100K unique users per 24h) and only
    apply to business-initiated conversations, not user-initiated ones
  - Pricing differs: user-initiated (service) conversations have $0 per-
    conversation fee; business-initiated ones carry an additional charge
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any

from ..base import BaseTool, ToolResult

log = logging.getLogger(__name__)

_TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


class TwilioTool(BaseTool):
    """Send SMS and WhatsApp messages via the Twilio REST API.

    Auth via constructor args or env vars ``TWILIO_ACCOUNT_SID``,
    ``TWILIO_AUTH_TOKEN``, ``TWILIO_FROM_NUMBER``.  Uses stdlib
    ``urllib`` only — no extra dependencies.

    Usage::

        tool = TwilioTool()
        result = await tool.run(action="send_sms", to="+15551234567", body="hello")
        result = await tool.run(action="send_whatsapp", to="+15551234567", body="hi")
    """

    name = "twilio"
    description = "Send SMS or WhatsApp messages via Twilio. Actions: send_sms, send_whatsapp."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform",
                "enum": ["send_sms", "send_whatsapp"],
            },
            "to": {
                "type": "string",
                "description": "Recipient phone number in E.164 format (e.g. +15551234567)",
            },
            "body": {
                "type": "string",
                "description": "Message body",
            },
        },
        "required": ["action", "to", "body"],
    }

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
    ) -> None:
        self._sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID", "")
        self._token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN", "")
        self._from = from_number or os.environ.get("TWILIO_FROM_NUMBER", "")

        log.warning(
            "TwilioTool instantiated — this tool can send messages to "
            "arbitrary phone numbers. Make sure this is intentional."
        )

    async def run(
        self, action: str = "", to: str = "", body: str = "", **kwargs: Any
    ) -> ToolResult:
        if not action:
            return ToolResult(output="", error="No action specified.")
        if not to:
            return ToolResult(output="", error="No recipient (to) provided.")
        if not body:
            return ToolResult(output="", error="No message body provided.")

        handlers = {
            "send_sms": self._send_sms,
            "send_whatsapp": self._send_whatsapp,
        }
        handler = handlers.get(action)
        if handler is None:
            return ToolResult(
                output="",
                error=f"Unknown action: {action}. Must be one of: {', '.join(handlers)}",
            )

        try:
            return await handler(to=to, body=body)
        except Exception as e:
            return ToolResult(output="", error=f"Twilio error: {e}")

    def _check_creds(self) -> str | None:
        if not self._sid or not self._token:
            return "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are required."
        if not self._from:
            return "TWILIO_FROM_NUMBER is required."
        return None

    async def _post(self, payload: dict[str, str]) -> dict[str, Any]:
        url = _TWILIO_API.format(sid=self._sid)
        data = urllib.parse.urlencode(payload).encode()

        creds = base64.b64encode(f"{self._sid}:{self._token}".encode()).decode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

        loop = asyncio.get_running_loop()

        def _fetch() -> dict[str, Any]:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())  # type: ignore[no-any-return]

        return await loop.run_in_executor(None, _fetch)

    async def _send_sms(self, to: str, body: str) -> ToolResult:
        err = self._check_creds()
        if err:
            return ToolResult(output="", error=err)

        resp = await self._post({"To": to, "From": self._from, "Body": body})
        sid = resp.get("sid", "unknown")
        return ToolResult(output=f"SMS sent to {to} (sid={sid}).")

    async def _send_whatsapp(self, to: str, body: str) -> ToolResult:
        err = self._check_creds()
        if err:
            return ToolResult(output="", error=err)

        wa_to = f"whatsapp:{to}" if not to.startswith("whatsapp:") else to
        wa_from = f"whatsapp:{self._from}" if not self._from.startswith("whatsapp:") else self._from

        resp = await self._post({"To": wa_to, "From": wa_from, "Body": body})
        sid = resp.get("sid", "unknown")
        return ToolResult(output=f"WhatsApp message sent to {to} (sid={sid}).")
