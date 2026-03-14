from __future__ import annotations

from .base import Document

class DiscordLoader:
    """Load messages from a Discord channel.
    
    Requires 'discord.py' package.
    """

    def __init__(self, bot_token: str, channel_id: int | str):
        self.bot_token = bot_token
        self.channel_id = str(channel_id)

    def load(self, limit: int = 100) -> list[Document]:
        """
        Sync: Load messages from Discord. 
        """
        import asyncio
        # We need a fresh event loop if one isn't running
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.load_async(limit=limit))

    async def load_async(self, limit: int = 100) -> list[Document]:
        """Async: Load messages from Discord."""
        try:
            import discord
        except ImportError:
            raise ImportError(
                "Discord loader requires 'discord.py' package. "
                "Install it with: pip install discord.py"
            )

        intents = discord.Intents.default()
        # We need message_content intent to read messages
        intents.message_content = True
        client = discord.Client(intents=intents)
        documents = []

        @client.event
        async def on_ready():
            try:
                channel = client.get_channel(int(self.channel_id))
                if not channel:
                    try:
                        channel = await client.fetch_channel(int(self.channel_id))
                    except Exception:
                        pass
                
                if not channel:
                    print(f"Error: Could not find channel {self.channel_id}")
                    return

                # Check if it's a channel we can read
                if hasattr(channel, 'history'):
                    async for message in channel.history(limit=limit):
                        if message.content:
                            documents.append(Document(
                                text=message.content,
                                metadata={
                                    "author": str(message.author),
                                    "timestamp": message.created_at.isoformat(),
                                    "channel_id": self.channel_id,
                                    "message_id": str(message.id),
                                    "attachments": [str(a.url) for a in message.attachments]
                                }
                            ))
            finally:
                await client.close()

        try:
            await client.start(self.bot_token)
        except Exception as e:
            # Ensure client is closed on error
            if not client.is_closed():
                await client.close()
            raise e

        return documents
