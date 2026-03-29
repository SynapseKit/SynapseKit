"""DiscordLoader — load messages from Discord channels via the Discord API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

from .base import Document


class DiscordLoader:
    """Load messages from Discord channels.

    Requires the `discord.py` package (install with `pip install discord.py`).

    Example::

        loader = DiscordLoader(
            token="your-bot-token",
            channel_ids=[123456789012345678],
            max_messages=100,
        )
        docs = loader.load()
    """

    def __init__(
        self,
        token: str,
        channel_ids: list[int],
        max_messages: Optional[int] = 100,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> None:
        self.token = token
        self.channel_ids = channel_ids
        self.max_messages = max_messages
        self.before = before
        self.after = after

    def load(self) -> list[Document]:
        """Synchronously fetch messages and return as Documents."""
        try:
            import discord
        except ImportError:
            raise ImportError(
                "discord.py is required for DiscordLoader. "
                "Install it with: pip install discord.py"
            ) from None

        # Run the async load method in a new event loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._aload())
        finally:
            loop.close()

    async def aload(self) -> list[Document]:
        """Async version of load."""
        return await self._aload()

    async def _aload(self) -> list[Document]:
        """Internal async implementation."""
        import discord

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        documents: list[Document] = []

        @client.event
        async def on_ready() -> None:
            try:
                for channel_id in self.channel_ids:
                    channel = client.get_channel(channel_id)
                    if channel is None:
                        try:
                            channel = await client.fetch_channel(channel_id)
                        except discord.NotFound:
                            print(f"Channel {channel_id} not found, skipping.")
                            continue
                        except discord.Forbidden:
                            print(f"No permission to access channel {channel_id}, skipping.")
                            continue

                    if not isinstance(channel, discord.TextChannel):
                        print(f"Channel {channel_id} is not a text channel, skipping.")
                        continue

                    # Build kwargs for history()
                    kwargs: dict[str, Any] = {"limit": self.max_messages}
                    if self.before:
                        kwargs["before"] = self.before
                    if self.after:
                        kwargs["after"] = self.after

                    async for message in channel.history(**kwargs):
                        if message.author.bot:
                            continue

                        content = message.clean_content.strip()
                        if not content:
                            continue

                        documents.append(
                            Document(
                                text=content,
                                metadata={
                                    "source": f"discord-channel-{channel_id}",
                                    "loader": "DiscordLoader",
                                    "message_id": message.id,
                                    "author": str(message.author),
                                    "author_id": message.author.id,
                                    "created_at": message.created_at.isoformat(),
                                    "channel_id": channel_id,
                                    "channel_name": channel.name,
                                },
                            )
                        )
            finally:
                await client.close()

        await client.start(self.token)
        return documents
