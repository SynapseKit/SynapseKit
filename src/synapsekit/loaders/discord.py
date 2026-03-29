from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from .base import Document

try:
    import discord
    from discord.ext import commands
except ImportError:
    discord = None
    commands = None

logger = logging.getLogger(__name__)


class DiscordLoader:
    """Load messages from Discord channels.

    This loader requires the `discord.py` library and a bot token.

    Usage::

        from synapsekit.loaders import DiscordLoader

        loader = DiscordLoader(token="YOUR_BOT_TOKEN")
        # Load messages from a specific channel
        docs = await loader.load_channel(channel_id=1234567890, limit=100)
        # Or load messages from multiple channels
        docs = await loader.load_channels(channel_ids=[1234567890, 9876543210], limit_per_channel=50)

    You can also filter messages by date range or specific authors.

    Note:
        The bot must have the necessary intents enabled (typically
        ``discord.Intents.default()`` with ``message_content=True``).
    """

    def __init__(
        self,
        token: str,
        *,
        intents: Optional[discord.Intents] = None,
        bot: Optional[discord.Client] = None,
    ) -> None:
        """Initialize the Discord loader.

        Args:
            token: Discord bot token.
            intents: Discord intents configuration. If not provided,
                uses ``discord.Intents.default()`` with ``message_content=True``.
            bot: Custom discord.Client instance. If provided, ``token`` and
                ``intents`` are ignored and the loader uses this client instead.
                The caller is responsible for starting/stopping the client.
        """
        if discord is None:
            raise ImportError(
                "discord.py is not installed. Install it with: pip install discord.py"
            )

        self.token = token
        self._bot = bot
        self._intents = intents

        if self._bot is None:
            if self._intents is None:
                self._intents = discord.Intents.default()
                self._intents.message_content = True
            self._bot = discord.Client(intents=self._intents)

        self._is_external_bot = bot is not None

    async def load_channel(
        self,
        channel_id: int,
        *,
        limit: Optional[int] = 100,
        before: Optional[datetime | discord.Object | int] = None,
        after: Optional[datetime | discord.Object | int] = None,
        around: Optional[datetime | discord.Object | int] = None,
        oldest_first: bool = False,
        author_id: Optional[int] = None,
        include_metadata: bool = True,
    ) -> list[Document]:
        """Load messages from a single Discord channel.

        Args:
            channel_id: ID of the channel to load messages from.
            limit: Maximum number of messages to retrieve (default 100).
                Pass ``None`` to retrieve all messages (use with caution).
            before: Retrieve messages before this date/message ID.
            after: Retrieve messages after this date/message ID.
            around: Retrieve messages around this date/message ID.
            oldest_first: If True, return messages in chronological order
                (oldest first). Default is False (newest first).
            author_id: If provided, only include messages from this user ID.
            include_metadata: If True (default), include message metadata
                (author, timestamp, etc.) in the Document metadata.

        Returns:
            List of Document objects, each containing message text and metadata.
        """
        if not self._is_external_bot:
            await self._start_bot_if_needed()

        channel = self._bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._bot.fetch_channel(channel_id)
            except discord.errors.Forbidden:
                logger.error(f"Missing permissions to access channel {channel_id}")
                return []
            except discord.errors.NotFound:
                logger.error(f"Channel {channel_id} not found")
                return []

        messages: list[discord.Message] = []
        async for msg in channel.history(
            limit=limit,
            before=before,
            after=after,
            around=around,
            oldest_first=oldest_first,
        ):
            if author_id is not None and msg.author.id != author_id:
                continue
            messages.append(msg)

        return self._messages_to_documents(messages, include_metadata)

    async def load_channels(
        self,
        channel_ids: list[int],
        *,
        limit_per_channel: Optional[int] = 50,
        **kwargs: Any,
    ) -> list[Document]:
        """Load messages from multiple Discord channels concurrently.

        Args:
            channel_ids: List of channel IDs to load from.
            limit_per_channel: Maximum messages per channel (default 50).
            **kwargs: Additional arguments passed to ``load_channel``.

        Returns:
            Combined list of Document objects from all channels.
        """
        tasks = [
            self.load_channel(cid, limit=limit_per_channel, **kwargs)
            for cid in channel_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: list[Document] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to load channel {channel_ids[i]}: {result}",
                    exc_info=result,
                )
                continue
            docs.extend(result)
        return docs

    async def load_guild(
        self,
        guild_id: int,
        *,
        channel_types: Optional[list[discord.ChannelType]] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Load messages from all text channels in a guild (server).

        Args:
            guild_id: ID of the guild to load from.
            channel_types: List of channel types to include.
                Default is ``[discord.ChannelType.text, discord.ChannelType.news]``.
            **kwargs: Additional arguments passed to ``load_channel``.

        Returns:
            Combined list of Document objects from all matching channels.
        """
        if not self._is_external_bot:
            await self._start_bot_if_needed()

        guild = self._bot.get_guild(guild_id)
        if guild is None:
            try:
                guild = await self._bot.fetch_guild(guild_id)
            except discord.errors.Forbidden:
                logger.error(f"Missing permissions to access guild {guild_id}")
                return []
            except discord.errors.NotFound:
                logger.error(f"Guild {guild_id} not found")
                return []

        if channel_types is None:
            channel_types = [discord.ChannelType.text, discord.ChannelType.news]

        channel_ids = [
            ch.id
            for ch in guild.channels
            if isinstance(ch, discord.TextChannel) and ch.type in channel_types
        ]

        return await self.load_channels(channel_ids, **kwargs)

    async def _start_bot_if_needed(self) -> None:
        """Start the bot client if it's not already running."""
        if not self._bot.is_ready():
            # Start the bot in the background
            asyncio.create_task(self._bot.start(self.token))
            # Wait for it to be ready
            await self._bot.wait_until_ready()

    def _messages_to_documents(
        self, messages: list[discord.Message], include_metadata: bool = True
    ) -> list[Document]:
        """Convert Discord messages to Document objects."""
        docs = []
        for msg in messages:
            metadata = {}
            if include_metadata:
                metadata = {
                    "id": msg.id,
                    "author_id": msg.author.id,
                    "author_name": str(msg.author),
                    "channel_id": msg.channel.id,
                    "channel_name": getattr(msg.channel, "name", None),
                    "created_at": msg.created_at.isoformat(),
                    "edited_at": msg.edited_at.isoformat() if msg.edited_at else None,
                    "type": str(msg.type),
                    "attachments": [
                        {
                            "filename": a.filename,
                            "url": a.url,
                            "size": a.size,
                        }
                        for a in msg.attachments
                    ],
                    "embeds": len(msg.embeds),
                    "reactions": [str(r) for r in msg.reactions],
                }
                # Add guild info if available
                if hasattr(msg.channel, "guild"):
                    metadata["guild_id"] = msg.channel.guild.id
                    metadata["guild_name"] = msg.channel.guild.name

            # Combine message content with any embeds/attachments text
            text_parts = [msg.clean_content]
            for embed in msg.embeds:
                if embed.title:
                    text_parts.append(f"Embed title: {embed.title}")
                if embed.description:
                    text_parts.append(embed.description)
                for field in embed.fields:
                    text_parts.append(f"{field.name}: {field.value}")

            full_text = "\n".join(filter(None, text_parts))
            docs.append(Document(text=full_text, metadata=metadata))
        return docs

    async def close(self) -> None:
        """Close the Discord client if it was created by this loader."""
        if not self._is_external_bot and self._bot.is_ready():
            await self._bot.close()
