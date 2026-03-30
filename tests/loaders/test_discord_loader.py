"""Tests for DiscordLoader."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.loaders.discord import DiscordLoader


class TestDiscordLoaderValidation:
    def test_channel_id_must_be_int(self):
        with pytest.raises(TypeError, match="channel_id must be an integer"):
            DiscordLoader(token="tok", channel_id="not-an-int")  # type: ignore[arg-type]

    def test_limit_must_be_positive(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            DiscordLoader(token="tok", channel_id=123, limit=0)

    def test_negative_limit_raises(self):
        with pytest.raises(ValueError):
            DiscordLoader(token="tok", channel_id=123, limit=-1)

    def test_valid_construction(self):
        loader = DiscordLoader(token="tok", channel_id=123456789)
        assert loader.channel_id == 123456789
        assert loader.limit == 100
        assert loader.include_metadata is True

    def test_custom_params(self):
        loader = DiscordLoader(
            token="tok",
            channel_id=123,
            limit=50,
            before_message_id=999,
            after_message_id=111,
            include_metadata=False,
        )
        assert loader.limit == 50
        assert loader.before_message_id == 999
        assert loader.after_message_id == 111
        assert loader.include_metadata is False


class TestDiscordLoaderImport:
    def test_missing_discord_raises_import_error(self):
        import sys

        loader = DiscordLoader(token="tok", channel_id=123)
        with patch.dict(sys.modules, {"discord": None}):
            with pytest.raises(ImportError, match=r"discord\.py is required"):
                loader.load()


class TestDiscordLoaderDocuments:
    def _make_mock_message(
        self, text: str, author: str = "user#1234", msg_id: int = 1
    ) -> MagicMock:
        msg = MagicMock()
        msg.clean_content = text
        msg.id = msg_id
        msg.author.id = 42
        msg.author.__str__ = lambda self: author
        msg.channel.id = 999
        msg.created_at.isoformat.return_value = "2026-01-01T00:00:00"
        msg.edited_at = None
        msg.attachments = []
        msg.reactions = []
        return msg

    def _make_discord_mock(self, messages: list) -> MagicMock:
        discord_mock = MagicMock()

        # Mock discord.Intents.default() to return something with message_content
        intents = MagicMock()
        discord_mock.Intents.default.return_value = intents

        # Mock discord.Object
        discord_mock.Object = MagicMock(side_effect=lambda id: MagicMock(id=id))

        # Build async channel.history generator
        async def fake_history(**kwargs):
            for msg in messages:
                yield msg

        channel = MagicMock()
        channel.history = fake_history

        # Mock client
        client = MagicMock()
        client.get_channel.return_value = channel

        # Make client.start() trigger on_ready then return
        async def fake_start(token):
            # Find and call the on_ready handler registered via @client.event
            handler = client.event.call_args[0][0]
            await handler()

        client.start = fake_start
        client.close = AsyncMock()

        discord_mock.Client.return_value = client
        return discord_mock

    async def test_aload_returns_documents(self):
        msgs = [
            self._make_mock_message("Hello world", msg_id=1),
            self._make_mock_message("Second message", msg_id=2),
        ]
        discord_mock = self._make_discord_mock(msgs)

        loader = DiscordLoader(token="tok", channel_id=123)
        with patch.dict("sys.modules", {"discord": discord_mock}):
            docs = await loader._aload_with_client()

        assert len(docs) == 2
        assert docs[0].text == "Hello world"
        assert docs[1].text == "Second message"

    async def test_aload_metadata_included_by_default(self):
        msg = self._make_mock_message("Test msg", msg_id=10)
        discord_mock = self._make_discord_mock([msg])

        loader = DiscordLoader(token="tok", channel_id=123)
        with patch.dict("sys.modules", {"discord": discord_mock}):
            docs = await loader._aload_with_client()

        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["source"] == "discord:123"
        assert meta["loader"] == "DiscordLoader"
        assert meta["message_id"] == 10
        assert meta["channel_id"] == 999
        assert "created_at" in meta

    async def test_aload_metadata_excluded_when_disabled(self):
        msg = self._make_mock_message("No meta")
        discord_mock = self._make_discord_mock([msg])

        loader = DiscordLoader(token="tok", channel_id=123, include_metadata=False)
        with patch.dict("sys.modules", {"discord": discord_mock}):
            docs = await loader._aload_with_client()

        assert docs[0].metadata == {}

    async def test_aload_empty_channel(self):
        discord_mock = self._make_discord_mock([])

        loader = DiscordLoader(token="tok", channel_id=123)
        with patch.dict("sys.modules", {"discord": discord_mock}):
            docs = await loader._aload_with_client()

        assert docs == []

    async def test_before_and_after_message_ids_passed_to_history(self):
        discord_mock = self._make_discord_mock([])
        obj_calls = []

        def capture_obj(id):
            obj_calls.append(id)
            return MagicMock(id=id)

        discord_mock.Object.side_effect = capture_obj

        loader = DiscordLoader(
            token="tok", channel_id=123, before_message_id=500, after_message_id=100
        )
        with patch.dict("sys.modules", {"discord": discord_mock}):
            await loader._aload_with_client()

        assert 500 in obj_calls
        assert 100 in obj_calls

    def test_load_sync_wraps_async(self):
        """load() should run the async path and return Documents."""
        from synapsekit.loaders.base import Document

        expected = [Document(text="sync msg", metadata={})]

        loader = DiscordLoader(token="tok", channel_id=123)

        discord_mock = MagicMock()

        with patch.dict("sys.modules", {"discord": discord_mock}):
            with patch.object(loader, "_aload_with_client", new=AsyncMock(return_value=expected)):
                result = loader.load()

        assert result == expected
