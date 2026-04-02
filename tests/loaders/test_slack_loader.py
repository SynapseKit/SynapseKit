"""Tests for SlackLoader."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.loaders.slack import SlackLoader


class TestSlackLoaderValidation:
    def test_load_sync_wraps_aload(self):
        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123")
        expected = [MagicMock()]
        with patch.object(loader, "aload", new=AsyncMock(return_value=expected)):
            result = loader.load()
        assert result == expected

    def test_valid_construction(self):
        loader = SlackLoader(bot_token="xoxb-test-token", channel_id="C123456")
        assert loader.bot_token == "xoxb-test-token"
        assert loader.channel_id == "C123456"
        assert loader.limit is None

    def test_custom_params(self):
        loader = SlackLoader(
            bot_token="xoxb-test",
            channel_id="C999",
            limit=50,
        )
        assert loader.limit == 50


class TestSlackLoaderImport:
    async def test_missing_slack_sdk_raises_import_error(self):
        import sys

        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123")
        with patch.dict(sys.modules, {"slack_sdk": None, "slack_sdk.web": None}):
            with pytest.raises(ImportError, match=r"slack-sdk required"):
                await loader.aload()


class TestSlackLoaderDocuments:
    def _make_mock_message(
        self,
        text: str,
        user: str = "U123456",
        ts: str = "1234567890.123456",
        thread_ts: str | None = None,
    ) -> dict:
        msg = {
            "text": text,
            "user": user,
            "ts": ts,
            "type": "message",
        }
        if thread_ts:
            msg["thread_ts"] = thread_ts
        return msg

    def _make_mock_client(
        self, messages: list[dict], thread_replies: dict[str, list[dict]] | None = None
    ) -> MagicMock:
        """Create a mock AsyncWebClient."""
        client = MagicMock()
        client.session = MagicMock()
        client.session.close = AsyncMock()

        # Mock conversations_history
        async def fake_history(**kwargs):
            response = {
                "ok": True,
                "messages": messages,
                "response_metadata": {},
            }
            return response

        client.conversations_history = AsyncMock(side_effect=fake_history)

        # Mock conversations_replies
        async def fake_replies(**kwargs):
            ts = kwargs.get("ts")
            replies = thread_replies.get(ts, []) if thread_replies else []
            # Slack API returns parent message + replies
            parent = next((m for m in messages if m.get("ts") == ts), None)
            all_messages = [parent, *replies] if parent else replies
            return {
                "ok": True,
                "messages": all_messages,
            }

        client.conversations_replies = AsyncMock(side_effect=fake_replies)

        return client

    async def test_load_returns_documents(self):
        msgs = [
            self._make_mock_message("Hello from Slack", ts="1000.0"),
            self._make_mock_message("Second message", ts="1001.0"),
        ]

        # Mock the slack_sdk module
        slack_sdk_mock = MagicMock()
        async_client_mock = MagicMock()
        mock_client_instance = self._make_mock_client(msgs)
        async_client_mock.AsyncWebClient = MagicMock(return_value=mock_client_instance)
        slack_sdk_mock.web.async_client = async_client_mock

        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123")

        with patch.dict(
            "sys.modules",
            {
                "slack_sdk": slack_sdk_mock,
                "slack_sdk.web": slack_sdk_mock.web,
                "slack_sdk.web.async_client": async_client_mock,
            },
        ):
            docs = await loader.aload()

        assert len(docs) == 2
        assert docs[0].text == "Hello from Slack"
        assert docs[1].text == "Second message"

    async def test_load_with_metadata(self):
        msgs = [
            self._make_mock_message("Test message", user="U999", ts="1234.5"),
        ]

        # Mock the slack_sdk module
        slack_sdk_mock = MagicMock()
        async_client_mock = MagicMock()
        mock_client_instance = self._make_mock_client(msgs)
        async_client_mock.AsyncWebClient = MagicMock(return_value=mock_client_instance)
        slack_sdk_mock.web.async_client = async_client_mock

        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123")

        with patch.dict(
            "sys.modules",
            {
                "slack_sdk": slack_sdk_mock,
                "slack_sdk.web": slack_sdk_mock.web,
                "slack_sdk.web.async_client": async_client_mock,
            },
        ):
            docs = await loader.aload()

        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["source"] == "slack"
        assert meta["channel"] == "C123"
        assert meta["user"] == "U999"
        assert meta["timestamp"] == "1234.5"
        assert meta["thread"] is False

    async def test_load_with_thread_replies(self):
        # Parent message with thread
        parent_ts = "1000.0"
        msgs = [
            self._make_mock_message("Parent message", ts=parent_ts, thread_ts=parent_ts),
        ]

        # Thread replies
        thread_replies = {
            parent_ts: [
                self._make_mock_message("Reply 1", ts="1000.1"),
                self._make_mock_message("Reply 2", ts="1000.2"),
            ]
        }

        # Mock the slack_sdk module
        slack_sdk_mock = MagicMock()
        async_client_mock = MagicMock()
        mock_client_instance = self._make_mock_client(msgs, thread_replies)
        async_client_mock.AsyncWebClient = MagicMock(return_value=mock_client_instance)
        slack_sdk_mock.web.async_client = async_client_mock

        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123")

        with patch.dict(
            "sys.modules",
            {
                "slack_sdk": slack_sdk_mock,
                "slack_sdk.web": slack_sdk_mock.web,
                "slack_sdk.web.async_client": async_client_mock,
            },
        ):
            docs = await loader.aload()

        assert len(docs) == 1
        assert "Parent message" in docs[0].text
        assert "[Thread replies:]" in docs[0].text
        assert "Reply 1" in docs[0].text
        assert "Reply 2" in docs[0].text
        assert docs[0].metadata["thread"] is True

    async def test_load_skips_empty_messages(self):
        msgs = [
            self._make_mock_message("Valid message", ts="1000.0"),
            self._make_mock_message("", ts="1001.0"),  # Empty
            self._make_mock_message("   ", ts="1002.0"),  # Whitespace only
            self._make_mock_message("Another valid", ts="1003.0"),
        ]

        # Mock the slack_sdk module
        slack_sdk_mock = MagicMock()
        async_client_mock = MagicMock()
        mock_client_instance = self._make_mock_client(msgs)
        async_client_mock.AsyncWebClient = MagicMock(return_value=mock_client_instance)
        slack_sdk_mock.web.async_client = async_client_mock

        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123")

        with patch.dict(
            "sys.modules",
            {
                "slack_sdk": slack_sdk_mock,
                "slack_sdk.web": slack_sdk_mock.web,
                "slack_sdk.web.async_client": async_client_mock,
            },
        ):
            docs = await loader.aload()

        assert len(docs) == 2
        assert docs[0].text == "Valid message"
        assert docs[1].text == "Another valid"

    async def test_load_with_limit(self):
        msgs = [self._make_mock_message(f"Message {i}", ts=f"{1000 + i}.0") for i in range(10)]

        # Mock the slack_sdk module
        slack_sdk_mock = MagicMock()
        async_client_mock = MagicMock()
        mock_client_instance = self._make_mock_client(msgs)
        async_client_mock.AsyncWebClient = MagicMock(return_value=mock_client_instance)
        slack_sdk_mock.web.async_client = async_client_mock

        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123", limit=5)

        with patch.dict(
            "sys.modules",
            {
                "slack_sdk": slack_sdk_mock,
                "slack_sdk.web": slack_sdk_mock.web,
                "slack_sdk.web.async_client": async_client_mock,
            },
        ):
            docs = await loader.aload()

        assert len(docs) == 5

    async def test_load_empty_channel(self):
        # Mock the slack_sdk module
        slack_sdk_mock = MagicMock()
        async_client_mock = MagicMock()
        mock_client_instance = self._make_mock_client([])
        async_client_mock.AsyncWebClient = MagicMock(return_value=mock_client_instance)
        slack_sdk_mock.web.async_client = async_client_mock

        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123")

        with patch.dict(
            "sys.modules",
            {
                "slack_sdk": slack_sdk_mock,
                "slack_sdk.web": slack_sdk_mock.web,
                "slack_sdk.web.async_client": async_client_mock,
            },
        ):
            docs = await loader.aload()

        assert docs == []

    async def test_pagination(self):
        """Test that pagination works with cursor."""
        first_batch = [
            self._make_mock_message("Message 1", ts="1000.0"),
            self._make_mock_message("Message 2", ts="1001.0"),
        ]
        second_batch = [
            self._make_mock_message("Message 3", ts="1002.0"),
            self._make_mock_message("Message 4", ts="1003.0"),
        ]

        call_count = 0

        async def fake_history(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "ok": True,
                    "messages": first_batch,
                    "response_metadata": {"next_cursor": "cursor_123"},
                }
            else:
                return {
                    "ok": True,
                    "messages": second_batch,
                    "response_metadata": {},
                }

        client = MagicMock()
        client.session = MagicMock()
        client.session.close = AsyncMock()
        client.conversations_history = AsyncMock(side_effect=fake_history)
        client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": []})

        # Mock the slack_sdk module
        slack_sdk_mock = MagicMock()
        async_client_mock = MagicMock()
        async_client_mock.AsyncWebClient = MagicMock(return_value=client)
        slack_sdk_mock.web.async_client = async_client_mock

        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123")

        with patch.dict(
            "sys.modules",
            {
                "slack_sdk": slack_sdk_mock,
                "slack_sdk.web": slack_sdk_mock.web,
                "slack_sdk.web.async_client": async_client_mock,
            },
        ):
            docs = await loader.aload()

        assert len(docs) == 4
        assert call_count == 2

    async def test_rate_limit_handling(self):
        """Test that rate limiting is handled with retry."""

        call_count = 0

        async def fake_history(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate rate limit error
                error = Exception("Rate limited")
                error.response = MagicMock()
                error.response.status_code = 429
                error.response.headers = {"Retry-After": "1"}
                raise error
            else:
                # Second call succeeds
                return {
                    "ok": True,
                    "messages": [self._make_mock_message("Success", ts="1000.0")],
                    "response_metadata": {},
                }

        client = MagicMock()
        client.session = MagicMock()
        client.session.close = AsyncMock()
        client.conversations_history = AsyncMock(side_effect=fake_history)
        client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": []})

        # Mock the slack_sdk module
        slack_sdk_mock = MagicMock()
        async_client_mock = MagicMock()
        async_client_mock.AsyncWebClient = MagicMock(return_value=client)
        slack_sdk_mock.web.async_client = async_client_mock

        loader = SlackLoader(bot_token="xoxb-test", channel_id="C123")

        with patch.dict(
            "sys.modules",
            {
                "slack_sdk": slack_sdk_mock,
                "slack_sdk.web": slack_sdk_mock.web,
                "slack_sdk.web.async_client": async_client_mock,
            },
        ):
            docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Success"
        assert call_count == 2  # Retried after rate limit
