"""Tests for TeamsLoader."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.loaders.teams import TeamsLoader


def _make_mock_response(
    status_code: int, json_data: dict | None = None, headers: dict | None = None
) -> MagicMock:
    """Create a mock httpx.Response object."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    response.text = ""
    if json_data is not None:
        response.json = MagicMock(return_value=json_data)
    return response


class TestTeamsLoaderValidation:
    """Test TeamsLoader initialization and validation."""

    def test_valid_construction(self):
        """Test basic initialization."""
        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )
        assert loader.access_token == "test-token"
        assert loader.team_id == "team-123"
        assert loader.channel_id == "channel-456"
        assert loader.limit is None

    def test_custom_params(self):
        """Test initialization with custom limit."""
        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
            limit=50,
        )
        assert loader.limit == 50

    def test_load_sync_wraps_aload(self):
        """Test that load() wraps aload() synchronously."""
        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )
        expected = []
        with patch.object(loader, "aload", new=AsyncMock(return_value=expected)):
            result = loader.load()
        assert result == expected


class TestTeamsLoaderImport:
    """Test import error handling."""

    @pytest.mark.asyncio
    async def test_missing_httpx_raises_import_error(self):
        """Test that missing httpx raises ImportError."""
        import sys

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )
        with patch.dict(sys.modules, {"httpx": None}):
            with pytest.raises(ImportError, match=r"httpx required"):
                await loader.aload()


class TestTeamsLoaderDocuments:
    """Test document creation from Teams messages."""

    @pytest.mark.asyncio
    async def test_load_returns_documents(self):
        """Test that messages are converted to documents."""
        messages = [
            {
                "id": "msg-1",
                "body": {"content": "<p>Hello from Teams</p>"},
                "from": {"user": {"displayName": "John Doe"}},
                "createdDateTime": "2024-01-01T10:00:00Z",
            },
            {
                "id": "msg-2",
                "body": {"content": "<p>Second message</p>"},
                "from": {"user": {"displayName": "Jane Smith"}},
                "createdDateTime": "2024-01-01T11:00:00Z",
            },
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 2
        assert docs[0].text == "Hello from Teams"
        assert docs[1].text == "Second message"

    @pytest.mark.asyncio
    async def test_load_with_metadata(self):
        """Test that metadata is correctly extracted."""
        messages = [
            {
                "id": "msg-123",
                "body": {"content": "<p>Test message</p>"},
                "from": {"user": {"displayName": "Test User"}},
                "createdDateTime": "2024-01-15T14:30:00Z",
            },
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-abc",
            channel_id="channel-xyz",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["source"] == "teams"
        assert meta["team_id"] == "team-abc"
        assert meta["channel_id"] == "channel-xyz"
        assert meta["message_id"] == "msg-123"
        assert meta["author"] == "Test User"
        assert meta["timestamp"] == "2024-01-15T14:30:00Z"

    @pytest.mark.asyncio
    async def test_load_skips_empty_messages(self):
        """Test that empty messages are skipped."""
        messages = [
            {"id": "msg-1", "body": {"content": "<p>Valid message</p>"}},
            {"id": "msg-2", "body": {"content": ""}},  # Empty content
            {"id": "msg-3", "body": {}},  # Missing content
            {"id": "msg-4", "body": {"content": "<p>   </p>"}},  # Whitespace only
            {"id": "msg-5", "body": {"content": "<p>Another valid</p>"}},
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 2
        assert docs[0].text == "Valid message"
        assert docs[1].text == "Another valid"


class TestTeamsLoaderHTMLStripping:
    """Test HTML to plain text conversion."""

    @pytest.mark.asyncio
    async def test_simple_html_tags(self):
        """Test stripping simple HTML tags."""
        messages = [
            {"id": "msg-1", "body": {"content": "<p>Hello <b>world</b></p>"}},
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_html_entities(self):
        """Test decoding HTML entities."""
        messages = [
            {"id": "msg-1", "body": {"content": "<p>Hello &amp; welcome</p>"}},
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs[0].text == "Hello & welcome"

    @pytest.mark.asyncio
    async def test_complex_html(self):
        """Test stripping complex HTML with multiple nested tags."""
        messages = [
            {
                "id": "msg-1",
                "body": {
                    "content": "<div><p>Hello <span style='color:red'>world</span>!</p><p>Second paragraph.</p></div>"
                },
            },
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs[0].text == "Hello world!Second paragraph."

    @pytest.mark.asyncio
    async def test_html_with_mentions(self):
        """Test handling HTML with user mentions."""
        messages = [
            {
                "id": "msg-1",
                "body": {"content": "<p>Hi <at>John Doe</at>, please review this.</p>"},
            },
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs[0].text == "Hi John Doe, please review this."


class TestTeamsLoaderPagination:
    """Test pagination handling."""

    @pytest.mark.asyncio
    async def test_pagination_with_next_link(self):
        """Test that pagination follows @odata.nextLink."""
        first_batch = [
            {"id": "msg-1", "body": {"content": "<p>Message 1</p>"}},
            {"id": "msg-2", "body": {"content": "<p>Message 2</p>"}},
        ]
        second_batch = [
            {"id": "msg-3", "body": {"content": "<p>Message 3</p>"}},
            {"id": "msg-4", "body": {"content": "<p>Message 4</p>"}},
        ]

        call_count = 0

        def create_response(url: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_mock_response(
                    200,
                    {
                        "value": first_batch,
                        "@odata.nextLink": "https://graph.microsoft.com/v1.0/next-page",
                    },
                )
            else:
                return _make_mock_response(200, {"value": second_batch})

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()

            async def mock_get(url: str, **kwargs):
                return create_response(url)

            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 4
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_pagination_respects_limit(self):
        """Test that limit stops pagination early."""
        first_batch = [
            {"id": "msg-1", "body": {"content": "<p>Message 1</p>"}},
            {"id": "msg-2", "body": {"content": "<p>Message 2</p>"}},
            {"id": "msg-3", "body": {"content": "<p>Message 3</p>"}},
        ]
        second_batch = [
            {"id": "msg-4", "body": {"content": "<p>Message 4</p>"}},
        ]

        call_count = 0

        def create_response(url: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_mock_response(
                    200,
                    {
                        "value": first_batch,
                        "@odata.nextLink": "https://graph.microsoft.com/v1.0/next-page",
                    },
                )
            else:
                return _make_mock_response(200, {"value": second_batch})

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
            limit=4,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()

            async def mock_get(url: str, **kwargs):
                return create_response(url)

            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 4
        assert call_count == 2  # Second page fetched but only 1 message used


class TestTeamsLoaderEdgeCases:
    """Test edge cases and missing fields."""

    @pytest.mark.asyncio
    async def test_empty_response(self):
        """Test handling empty response."""
        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": []}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs == []

    @pytest.mark.asyncio
    async def test_missing_author_uses_unknown(self):
        """Test that missing author defaults to 'unknown'."""
        messages = [
            {
                "id": "msg-1",
                "body": {"content": "<p>Test</p>"},
                "from": {},  # Missing user
            },
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs[0].metadata["author"] == "unknown"

    @pytest.mark.asyncio
    async def test_missing_display_name_uses_unknown(self):
        """Test that missing displayName defaults to 'unknown'."""
        messages = [
            {
                "id": "msg-1",
                "body": {"content": "<p>Test</p>"},
                "from": {"user": {}},  # Missing displayName
            },
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs[0].metadata["author"] == "unknown"

    @pytest.mark.asyncio
    async def test_missing_timestamp_empty_string(self):
        """Test that missing timestamp is empty string."""
        messages = [
            {
                "id": "msg-1",
                "body": {"content": "<p>Test</p>"},
                # No createdDateTime
            },
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs[0].metadata["timestamp"] == ""

    @pytest.mark.asyncio
    async def test_missing_body_handled(self):
        """Test that missing body is handled gracefully."""
        messages = [
            {"id": "msg-1"},  # No body at all
            {"id": "msg-2", "body": {"content": "<p>Valid</p>"}},
        ]

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=_make_mock_response(200, {"value": messages}))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Valid"


class TestTeamsLoaderRetry:
    """Test retry logic for rate limiting and server errors."""

    @pytest.mark.asyncio
    async def test_rate_limit_retry(self):
        """Test retry on 429 rate limit response."""
        call_count = 0

        def create_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_mock_response(429, headers={"Retry-After": "0"})
            else:
                return _make_mock_response(
                    200,
                    {"value": [{"id": "msg-1", "body": {"content": "<p>Success</p>"}}]},
                )

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=create_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Success"
        assert call_count == 2  # Retried once

    @pytest.mark.asyncio
    async def test_server_error_retry(self):
        """Test retry on 5xx server error."""
        call_count = 0

        def create_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_mock_response(503)
            else:
                return _make_mock_response(
                    200,
                    {"value": [{"id": "msg-1", "body": {"content": "<p>Success</p>"}}]},
                )

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=create_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert len(docs) == 1
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_auth_error_no_retry(self):
        """Test that 401 errors are not retried."""
        call_count = 0

        def create_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = _make_mock_response(401)
            response.text = "Unauthorized"
            return response

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=create_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs == []
        assert call_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_forbidden_error_no_retry(self):
        """Test that 403 errors are not retried."""
        call_count = 0

        def create_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = _make_mock_response(403)
            response.text = "Forbidden"
            return response

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=create_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs == []
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_not_found_error_no_retry(self):
        """Test that 404 errors are not retried."""
        call_count = 0

        def create_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            response = _make_mock_response(404)
            response.text = "Not Found"
            return response

        loader = TeamsLoader(
            access_token="test-token",
            team_id="team-123",
            channel_id="channel-456",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=create_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            docs = await loader.aload()

        assert docs == []
        assert call_count == 1
