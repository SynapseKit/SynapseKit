"""Email Tool: send emails via SMTP."""

from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class EmailTool(BaseTool):
    """Send an email via SMTP.

    SMTP configuration can be provided in the constructor or via environment
    variables: ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USER``, ``SMTP_PASSWORD``,
    ``SMTP_FROM``.

    Usage::

        tool = EmailTool(smtp_host="smtp.gmail.com", smtp_port=587,
                         smtp_user="me@gmail.com", smtp_password="app-pw",
                         from_addr="me@gmail.com")
        result = await tool.run(to="bob@example.com",
                                subject="Hello", body="Hi Bob!")
    """

    name = "send_email"
    description = "Send an email via SMTP."
    parameters = {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient email address",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line",
            },
            "body": {
                "type": "string",
                "description": "Email body text",
            },
        },
        "required": ["to", "subject", "body"],
    }

    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        from_addr: str | None = None,
    ) -> None:
        self._smtp_host = smtp_host or os.environ.get("SMTP_HOST")
        self._smtp_port = smtp_port or int(os.environ.get("SMTP_PORT", "587"))
        self._smtp_user = smtp_user or os.environ.get("SMTP_USER")
        self._smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD")
        self._from_addr = from_addr or os.environ.get("SMTP_FROM")

    async def run(
        self, to: str = "", subject: str = "", body: str = "", **kwargs: Any
    ) -> ToolResult:
        if not to or not subject or not body:
            return ToolResult(output="", error="'to', 'subject', and 'body' are all required.")

        if not self._smtp_host or not self._smtp_user or not self._smtp_password:
            return ToolResult(
                output="",
                error=(
                    "SMTP not configured. Provide smtp_host, smtp_user, and smtp_password "
                    "via constructor or SMTP_HOST/SMTP_USER/SMTP_PASSWORD env vars."
                ),
            )

        try:
            import asyncio
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self._from_addr or self._smtp_user
            msg["To"] = to

            loop = asyncio.get_event_loop()

            def _send():
                with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                    server.starttls()
                    server.login(self._smtp_user, self._smtp_password)
                    server.send_message(msg)

            await loop.run_in_executor(None, _send)

            return ToolResult(output=f"Email sent successfully to {to}.")
        except Exception as e:
            return ToolResult(output="", error=f"Failed to send email: {e}")
