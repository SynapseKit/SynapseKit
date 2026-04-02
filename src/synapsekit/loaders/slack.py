"""SlackLoader — load messages from Slack channels via the Slack API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import Document

logger = logging.getLogger(__name__)


class SlackLoader:
    """Load messages from Slack channels into Documents.

    This loader uses the Slack API to fetch messages from a specified channel,
    including thread replies. It supports both synchronous and asynchronous loading.

    Prerequisites:
        - A Slack bot token with appropriate permissions (channels:history, channels:read)
        - The bot must be added to the channel you want to read from

    Example::

        loader = SlackLoader(
            bot_token="xoxb-your-bot-token",
            channel_id="C123456789",
            limit=100,
        )
        docs = loader.load()        # synchronous
        # or
        docs = await loader.aload()  # asynchronous
    """

    def __init__(
        self,
        bot_token: str,
        channel_id: str,
        limit: int | None = None,
    ) -> None:
        self.bot_token = bot_token
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
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError:
            raise ImportError("slack-sdk required: pip install synapsekit[slack]") from None

        client = AsyncWebClient(token=self.bot_token)
        try:
            messages = await self._fetch_messages(client)

            documents = []
            for msg in messages:
                text = msg.get("text", "").strip()
                if not text:
                    continue

                # thread_ts == ts means this message is the thread parent
                thread_ts = msg.get("thread_ts")
                if thread_ts and thread_ts == msg.get("ts"):
                    replies = await self._fetch_thread_replies(client, self.channel_id, thread_ts)
                    if replies:
                        thread_text = "\n\n".join(
                            reply.get("text", "").strip()
                            for reply in replies
                            if reply.get("text", "").strip()
                        )
                        if thread_text:
                            text = f"{text}\n\n[Thread replies:]\n{thread_text}"

                metadata = {
                    "source": "slack",
                    "channel": self.channel_id,
                    "user": msg.get("user", "unknown"),
                    "timestamp": msg.get("ts", ""),
                    "thread": bool(thread_ts and thread_ts == msg.get("ts")),
                }

                documents.append(Document(text=text, metadata=metadata))

            return documents
        finally:
            await client.session.close()

    async def _fetch_messages(self, client: Any) -> list[dict]:
        """Fetch messages from the channel with pagination."""
        messages: list[dict] = []
        cursor = None
        fetched_count = 0
        max_rate_retries = 5

        while True:
            kwargs: dict[str, Any] = {"channel": self.channel_id, "limit": 100}
            if cursor:
                kwargs["cursor"] = cursor

            rate_retries = 0
            while True:
                try:
                    response = await client.conversations_history(**kwargs)
                    break
                except Exception as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status == 429 and rate_retries < max_rate_retries:
                        retry_after = int(getattr(e.response, "headers", {}).get("Retry-After", 1))
                        logger.warning(
                            "SlackLoader: rate limited, retrying in %ds (attempt %d/%d)",
                            retry_after,
                            rate_retries + 1,
                            max_rate_retries,
                        )
                        await asyncio.sleep(retry_after)
                        rate_retries += 1
                    else:
                        raise

            if not response["ok"]:
                break

            batch = response.get("messages", [])
            messages.extend(batch)
            fetched_count += len(batch)

            if self.limit and fetched_count >= self.limit:
                messages = messages[: self.limit]
                break

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return messages

    async def _fetch_thread_replies(
        self, client: Any, channel_id: str, thread_ts: str
    ) -> list[dict]:
        """Fetch replies to a thread.

        Slack's conversations.replies returns [parent, *replies]; we drop the parent.
        """
        try:
            response = await client.conversations_replies(channel=channel_id, ts=thread_ts)

            if not response["ok"]:
                return []

            messages = response.get("messages", [])
            # Index 0 is the parent message; skip it
            return messages[1:] if len(messages) > 1 else []

        except Exception as e:
            logger.warning(
                "SlackLoader: failed to fetch thread replies for ts=%s: %s",
                thread_ts,
                e,
            )
            return []
