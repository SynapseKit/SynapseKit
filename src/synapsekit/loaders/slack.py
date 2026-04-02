"""SlackLoader — load messages from Slack channels via the Slack API."""

from __future__ import annotations

import asyncio

from .base import Document


class SlackLoader:
    """Load messages from Slack channels into Documents.

    This loader uses the Slack API to fetch messages from a specified channel,
    including thread replies. It supports async loading and handles pagination.

    Prerequisites:
        - A Slack bot token with appropriate permissions (channels:history, channels:read)
        - The bot must be added to the channel you want to read from

    Example::

        loader = SlackLoader(
            bot_token="xoxb-your-bot-token",
            channel_id="C123456789",
            limit=100,
        )
        docs = await loader.load()
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

    async def load(self) -> list[Document]:
        """Asynchronously fetch messages and return them as Documents."""
        try:
            from slack_sdk.web.async_client import AsyncWebClient
        except ImportError:
            raise ImportError(
                "slack-sdk required: pip install synapsekit[slack]"
            ) from None

        client = AsyncWebClient(token=self.bot_token)
        messages = await self._fetch_messages(client)

        documents = []
        for msg in messages:
            # Skip messages without text (like join/leave notifications)
            text = msg.get("text", "").strip()
            if not text:
                continue

            # Fetch thread replies if this message has a thread
            thread_ts = msg.get("thread_ts")
            if thread_ts and thread_ts == msg.get("ts"):
                # This is a parent message with replies
                replies = await self._fetch_thread_replies(
                    client, self.channel_id, thread_ts
                )
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

    async def _fetch_messages(self, client) -> list[dict]:
        """Fetch messages from the channel with pagination."""
        messages = []
        cursor = None
        fetched_count = 0

        while True:
            try:
                # Build request parameters
                kwargs = {"channel": self.channel_id, "limit": 100}
                if cursor:
                    kwargs["cursor"] = cursor

                response = await client.conversations_history(**kwargs)

                if not response["ok"]:
                    break

                batch = response.get("messages", [])
                messages.extend(batch)
                fetched_count += len(batch)

                # Check if we've reached the limit
                if self.limit and fetched_count >= self.limit:
                    messages = messages[: self.limit]
                    break

                # Check for pagination cursor
                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            except Exception as e:
                # Handle rate limiting
                if hasattr(e, "response") and e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 1))
                    await asyncio.sleep(retry_after)
                    continue
                raise

        return messages

    async def _fetch_thread_replies(
        self, client, channel_id: str, thread_ts: str
    ) -> list[dict]:
        """Fetch replies to a thread."""
        try:
            response = await client.conversations_replies(
                channel=channel_id, ts=thread_ts
            )

            if not response["ok"]:
                return []

            # Skip the first message (parent) and return only replies
            messages = response.get("messages", [])
            return messages[1:] if len(messages) > 1 else []

        except Exception as e:
            # Handle rate limiting
            if hasattr(e, "response") and e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", 1))
                await asyncio.sleep(retry_after)
                # Retry once
                response = await client.conversations_replies(
                    channel=channel_id, ts=thread_ts
                )
                messages = response.get("messages", [])
                return messages[1:] if len(messages) > 1 else []
            # Silently skip thread replies on error
            return []
