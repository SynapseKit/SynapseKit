"""DiscordLoader — load messages from Discord channels via the Discord API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

from .base import Document


class DiscordLoader:
    """Load messages from Discord channels into Documents.

    This loader uses the Discord API to fetch messages from a specified channel.
    It supports both synchronous and asynchronous loading.

    Prerequisites:
        - A Discord bot token with appropriate permissions (read messages)
        - The bot must be added to the server containing the target channel

    Example::

        loader = DiscordLoader(
            token="your-bot-token",
            channel_id=123456789012345678,
            limit=100,
            before_message_id=None,
            after_message_id=None,
        )
        docs = loader.load()  # synchronous
        # or
        docs = await loader.aload()  # asynchronous
    """

    def __init__(
        self,
        token: str,
        channel_id: int,
        limit: int = 100,
        before_message_id: Optional[int] = None,
        after_message_id: Optional[int] = None,
        include_metadata: bool = True,
    ) -> None:
        self.token = token
        self.channel_id = channel_id
        self.limit = limit
        self.before_message_id = before_message_id
        self.after_message_id = after_message_id
        self.include_metadata = include_metadata

        if not isinstance(channel_id, int):
            raise TypeError(f"channel_id must be an integer, got {type(channel_id)}")
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

    def load(self) -> list[Document]:
        """Synchronously fetch messages and return them as Documents."""
        try:
            import discord
        except ImportError:
            raise ImportError(
                "discord.py is required for DiscordLoader. "
                "Install it with: pip install 'discord.py'"
            ) from None

        # Run the async load method in a new event loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._aload_with_client())
        finally:
            loop.close()

    async def aload(self) -> list[Document]:
        """Asynchronously fetch messages and return them as Documents."""
        return await self._aload_with_client()

    async def _aload_with_client(self) -> list[Document]:
        """Internal async method that uses discord.Client."""
        import discord

        intents = discord.Intents.default()
        intents.message_content = True

        client = discord.Client(intents=intents)
        documents = []

        @client.event
        async def on_ready():
            try:
                channel = client.get_channel(self.channel_id)
                if channel is None:
                    channel = await client.fetch_channel(self.channel_id)

                # Prepare kwargs for history fetching
                kwargs: dict[str, Any] = {"limit": self.limit}
                if self.before_message_id is not None:
                    kwargs["before"] = discord.Object(id=self.before_message_id)
                if self.after_message_id is not None:
                    kwargs["after"] = discord.Object(id=self.after_message_id)

                messages = [msg async for msg in channel.history(**kwargs)]

                for msg in messages:
                    metadata = {
                        "source": f"discord:{self.channel_id}",
                        "loader": "DiscordLoader",
                        "author": str(msg.author),
                        "author_id": msg.author.id,
                        "message_id": msg.id,
                        "channel_id": msg.channel.id,
                        "created_at": msg.created_at.isoformat(),
                        "edited_at": msg.edited_at.isoformat() if msg.edited_at else None,
                        "attachments": [
                            {
                                "filename": a.filename,
                                "url": a.url,
                                "size": a.size,
                            }
                            for a in msg.attachments
                        ],
                        "reactions": [
                            {
                                "emoji": str(r.emoji),
                                "count": r.count,
                            }
                            for r in msg.reactions
                        ],
                    } if self.include_metadata else {}

                    documents.append(
                        Document(
                            text=msg.clean_content,
                            metadata=metadata,
                        )
                    )
            finally:
                await client.close()

        await client.start(self.token)
        return documents
