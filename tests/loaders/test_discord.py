import pytest
from synapsekit.loaders.discord import DiscordLoader

def test_discord_loader_init():
    loader = DiscordLoader("token", "12345")
    assert loader.bot_token == "token"
    assert loader.channel_id == "12345"

@pytest.mark.asyncio
async def test_discord_loader_async_init():
    loader = DiscordLoader("token", "12345")
    assert loader.bot_token == "token"
    assert loader.channel_id == "12345"
