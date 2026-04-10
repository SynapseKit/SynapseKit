"""TeamsLoader — load messages from Microsoft Teams channels via Microsoft Graph API."""

from __future__ import annotations

import asyncio
import html
import logging
import re
from typing import TYPE_CHECKING, Any

from .base import Document

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

_GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


class TeamsLoader:
    """Load messages from Microsoft Teams channels into Documents.

    This loader uses the Microsoft Graph API to fetch messages from a specified
    Teams channel. It handles pagination automatically and converts HTML message
    content to plain text.

    Prerequisites:
        - A valid Microsoft Graph OAuth access token
        - The token must have Channels.Read.All or similar permissions

    Example::

        loader = TeamsLoader(
            access_token="eyJ0eXAiOiJKV1QiLCJhbGc...",
            team_id="02bd9fd6-8f93-4758-87c3-1e7375baf356",
            channel_id="19:09fc54a3141a45d0bc769cf506d2e079@thread.skype",
            limit=100,
        )
        docs = loader.load()        # synchronous
        # or
        docs = await loader.aload()  # asynchronous
    """

    def __init__(
        self,
        access_token: str,
        team_id: str,
        channel_id: str,
        limit: int | None = None,
    ) -> None:
        self.access_token = access_token
        self.team_id = team_id
        self.channel_id = channel_id
        self.limit = limit

    def load(self) -> list[Document]:
        """Synchronously fetch messages and return them as Documents."""
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.aload())
        finally:
            loop.close()

    async def aload(self) -> list[Document]:
        """Asynchronously fetch messages and return them as Documents."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install synapsekit[teams]") from None

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

        messages: list[dict[str, Any]] = []
        url = f"{_GRAPH_BASE_URL}/teams/{self.team_id}/channels/{self.channel_id}/messages"

        async with httpx.AsyncClient() as client:
            while url:
                response = await self._request_with_retry(client, url, headers)
                if response is None:
                    break

                data = response.json()
                batch = data.get("value", [])
                messages.extend(batch)

                if self.limit and len(messages) >= self.limit:
                    messages = messages[: self.limit]
                    break

                # Handle pagination via @odata.nextLink
                url = data.get("@odata.nextLink")

        documents = []
        for msg in messages:
            text = self._extract_text(msg)
            if not text:
                continue

            metadata = self._extract_metadata(msg)
            documents.append(Document(text=text, metadata=metadata))

        return documents

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
    ) -> httpx.Response | None:
        """Make HTTP request with exponential backoff retry for 429/5xx errors."""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = await client.get(url, headers=headers, timeout=30.0)

                if response.status_code == 200:
                    return response

                # Don't retry for auth/permission errors
                if response.status_code in (401, 403, 404):
                    logger.warning(
                        "TeamsLoader: request failed with status %d - %s",
                        response.status_code,
                        response.text[:200] if response.text else "no content",
                    )
                    return None

                # Retry for 429 (rate limit) or 5xx (server error)
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        delay = base_delay * (2**attempt)

                    logger.warning(
                        "TeamsLoader: rate limited or server error (status %d), retrying in %.1fs (attempt %d/%d)",
                        response.status_code,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue

                # Other errors - don't retry
                logger.warning(
                    "TeamsLoader: request failed with status %d - %s",
                    response.status_code,
                    response.text[:200] if response.text else "no content",
                )
                return None

            except Exception as e:
                # Handle httpx.RequestError and similar network errors
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "TeamsLoader: request error %s, retrying in %.1fs (attempt %d/%d)",
                        e,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("TeamsLoader: request failed after %d retries: %s", max_retries, e)
                    return None

        return None

    def _extract_text(self, message: dict[str, Any]) -> str:
        """Extract and clean text from message body.

        Microsoft Teams messages are in HTML format. This method extracts
        the body.content field and strips HTML tags to produce plain text.
        """
        body = message.get("body", {})
        if not body:
            return ""

        content = body.get("content", "")
        if not content:
            return ""

        # Decode HTML entities first (e.g., &nbsp; → space)
        text = html.unescape(content)

        # Remove HTML tags using simple regex
        # This handles common cases like <p>, <b>, <span>, etc.
        text = re.sub(r"<[^>]+>", "", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _extract_metadata(self, message: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata from a Teams message."""
        # Extract author
        author = "unknown"
        from_user = message.get("from", {})
        if from_user:
            user_info = from_user.get("user", {})
            if user_info:
                author = user_info.get("displayName", "unknown")

        # Extract timestamp
        timestamp = message.get("createdDateTime", "")

        return {
            "source": "teams",
            "team_id": self.team_id,
            "channel_id": self.channel_id,
            "message_id": message.get("id", ""),
            "author": author,
            "timestamp": timestamp,
        }
