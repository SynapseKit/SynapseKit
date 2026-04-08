"""Tests for JiraLoader."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synapsekit.loaders.jira import JiraLoader
from synapsekit.loaders.base import Document


def _make_mock_response(json_data=None, status_code=200):
    """Create a mock HTTP response."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.headers.get = MagicMock(return_value="2")
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        from httpx import HTTPStatusError, Request
        response.raise_for_status.side_effect = HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(spec=Request),
            response=response,
        )
    return response


@pytest.fixture
def jira_loader():
    """Create a JiraLoader instance for testing."""
    return JiraLoader(
        url="https://test.atlassian.net",
        username="test@example.com",
        api_token="test-token",
        jql="project = TEST",
    )


@pytest.fixture
def mock_issue_simple():
    """Mock a simple Jira issue with plain text."""
    return {
        "key": "TEST-123",
        "fields": {
            "summary": "Test issue summary",
            "description": "This is a plain text description",
            "status": {"name": "In Progress"},
            "assignee": {"displayName": "John Doe"},
            "priority": {"name": "High"},
            "comment": {
                "comments": [
                    {"body": "First comment"},
                    {"body": "Second comment"},
                ]
            },
        },
    }


@pytest.fixture
def mock_issue_adf():
    """Mock a Jira issue with ADF (Atlassian Document Format)."""
    return {
        "key": "TEST-456",
        "fields": {
            "summary": "ADF test issue",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {"type": "text", "text": "This is "},
                            {"type": "text", "text": "ADF formatted", "marks": [{"type": "strong"}]},
                            {"type": "text", "text": " text."},
                        ],
                    },
                    {
                        "type": "paragraph",
                        "content": [
                            {"type": "text", "text": "Second paragraph."},
                        ],
                    },
                ],
            },
            "status": {"name": "Done"},
            "assignee": None,
            "priority": None,
            "comment": {
                "comments": [
                    {
                        "body": {
                            "type": "doc",
                            "version": 1,
                            "content": [
                                {
                                    "type": "paragraph",
                                    "content": [{"type": "text", "text": "ADF comment"}],
                                }
                            ],
                        }
                    }
                ]
            },
        },
    }


@pytest.fixture
def mock_issue_minimal():
    """Mock a minimal Jira issue with missing fields."""
    return {
        "key": "TEST-789",
        "fields": {
            "summary": "Minimal issue",
            "description": None,
            "status": {"name": "Open"},
            "assignee": None,
            "priority": None,
            "comment": None,
        },
    }


class TestJiraLoaderInit:
    """Test JiraLoader initialization."""

    def test_init(self):
        """Test loader initialization with all parameters."""
        loader = JiraLoader(
            url="https://test.atlassian.net",
            username="user@example.com",
            api_token="token123",
            jql="project = MYPROJ",
            limit=10,
        )

        assert loader._url == "https://test.atlassian.net"
        assert loader._username == "user@example.com"
        assert loader._api_token == "token123"
        assert loader._jql == "project = MYPROJ"
        assert loader._limit == 10

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from URL."""
        loader = JiraLoader(
            url="https://test.atlassian.net/",
            username="user@example.com",
            api_token="token123",
            jql="project = TEST",
        )

        assert loader._url == "https://test.atlassian.net"


class TestADFExtraction:
    """Test ADF (Atlassian Document Format) text extraction."""

    def test_extract_text_from_plain_string(self, jira_loader):
        """Test extracting text from plain string."""
        result = jira_loader._extract_text_from_adf("Plain text")
        assert result == "Plain text"

    def test_extract_text_from_simple_adf(self, jira_loader):
        """Test extracting text from simple ADF node."""
        adf = {
            "type": "doc",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        }
        result = jira_loader._extract_text_from_adf(adf)
        assert "Hello" in result
        assert "World" in result

    def test_extract_text_from_nested_adf(self, jira_loader):
        """Test extracting text from nested ADF structure."""
        adf = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "First"},
                        {"type": "text", "text": "Second"},
                    ],
                },
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "Third"},
                    ],
                },
            ],
        }
        result = jira_loader._extract_text_from_adf(adf)
        assert "First" in result
        assert "Second" in result
        assert "Third" in result

    def test_extract_text_from_invalid_input(self, jira_loader):
        """Test extracting text from invalid input returns empty string."""
        assert jira_loader._extract_text_from_adf(None) == ""
        assert jira_loader._extract_text_from_adf(123) == ""
        assert jira_loader._extract_text_from_adf([]) == ""


class TestDescriptionExtraction:
    """Test description text extraction."""

    def test_extract_plain_text_description(self, jira_loader):
        """Test extracting plain text description."""
        result = jira_loader._extract_description("Plain text description")
        assert result == "Plain text description"

    def test_extract_adf_description(self, jira_loader):
        """Test extracting ADF formatted description."""
        adf = {
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "ADF text"}]},
            ],
        }
        result = jira_loader._extract_description(adf)
        assert "ADF text" in result

    def test_extract_none_description(self, jira_loader):
        """Test extracting None description returns empty string."""
        result = jira_loader._extract_description(None)
        assert result == ""


class TestCommentExtraction:
    """Test comment extraction."""

    def test_extract_comments_from_dict(self, jira_loader):
        """Test extracting comments from dict structure."""
        comment_field = {
            "comments": [
                {"body": "First comment"},
                {"body": "Second comment"},
            ]
        }
        result = jira_loader._extract_comments(comment_field)
        assert result == ["First comment", "Second comment"]

    def test_extract_comments_from_list(self, jira_loader):
        """Test extracting comments from list structure."""
        comment_field = [
            {"body": "Comment one"},
            {"body": "Comment two"},
        ]
        result = jira_loader._extract_comments(comment_field)
        assert result == ["Comment one", "Comment two"]

    def test_extract_adf_comments(self, jira_loader):
        """Test extracting ADF formatted comments."""
        comment_field = {
            "comments": [
                {
                    "body": {
                        "type": "doc",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": "ADF comment"}]},
                        ],
                    }
                }
            ]
        }
        result = jira_loader._extract_comments(comment_field)
        assert len(result) == 1
        assert "ADF comment" in result[0]

    def test_extract_comments_none(self, jira_loader):
        """Test extracting comments from None returns empty list."""
        result = jira_loader._extract_comments(None)
        assert result == []

    def test_extract_comments_empty(self, jira_loader):
        """Test extracting comments from empty dict returns empty list."""
        result = jira_loader._extract_comments({})
        assert result == []


class TestIssueToDocument:
    """Test converting Jira issues to Documents."""

    def test_simple_issue_conversion(self, jira_loader, mock_issue_simple):
        """Test converting a simple issue to Document."""
        doc = jira_loader._issue_to_document(mock_issue_simple)

        assert isinstance(doc, Document)
        assert "Test issue summary" in doc.text
        assert "plain text description" in doc.text
        assert "First comment" in doc.text
        assert "Second comment" in doc.text

        assert doc.metadata["source"] == "jira"
        assert doc.metadata["key"] == "TEST-123"
        assert doc.metadata["status"] == "In Progress"
        assert doc.metadata["assignee"] == "John Doe"
        assert doc.metadata["priority"] == "High"

    def test_adf_issue_conversion(self, jira_loader, mock_issue_adf):
        """Test converting an ADF formatted issue to Document."""
        doc = jira_loader._issue_to_document(mock_issue_adf)

        assert isinstance(doc, Document)
        assert "ADF test issue" in doc.text
        assert "ADF formatted" in doc.text
        assert "Second paragraph" in doc.text
        assert "ADF comment" in doc.text

        assert doc.metadata["source"] == "jira"
        assert doc.metadata["key"] == "TEST-456"
        assert doc.metadata["status"] == "Done"
        assert "assignee" not in doc.metadata
        assert "priority" not in doc.metadata

    def test_minimal_issue_conversion(self, jira_loader, mock_issue_minimal):
        """Test converting a minimal issue with missing fields."""
        doc = jira_loader._issue_to_document(mock_issue_minimal)

        assert isinstance(doc, Document)
        assert "Minimal issue" in doc.text

        assert doc.metadata["source"] == "jira"
        assert doc.metadata["key"] == "TEST-789"
        assert doc.metadata["status"] == "Open"
        assert "assignee" not in doc.metadata
        assert "priority" not in doc.metadata


class TestAsyncLoad:
    """Test async loading of Jira issues."""

    @pytest.mark.asyncio
    async def test_aload_single_page(self, jira_loader, mock_issue_simple):
        """Test loading issues from a single page."""
        mock_response_data = {
            "issues": [mock_issue_simple],
            "isLast": True,
        }

        mock_response = _make_mock_response(json_data=mock_response_data)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            documents = await jira_loader.aload()

            assert len(documents) == 1
            assert documents[0].metadata["key"] == "TEST-123"
            assert "Test issue summary" in documents[0].text

    @pytest.mark.asyncio
    async def test_aload_pagination(self, jira_loader, mock_issue_simple, mock_issue_adf):
        """Test loading issues with pagination."""
        mock_response_page1_data = {
            "issues": [mock_issue_simple],
            "isLast": False,
            "nextPageToken": "token123",
        }
        mock_response_page2_data = {
            "issues": [mock_issue_adf],
            "isLast": True,
        }

        mock_response1 = _make_mock_response(json_data=mock_response_page1_data)
        mock_response2 = _make_mock_response(json_data=mock_response_page2_data)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=[mock_response1, mock_response2])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            documents = await jira_loader.aload()

            assert len(documents) == 2
            assert documents[0].metadata["key"] == "TEST-123"
            assert documents[1].metadata["key"] == "TEST-456"
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_aload_with_limit(self, mock_issue_simple, mock_issue_adf):
        """Test loading issues with limit parameter."""
        loader = JiraLoader(
            url="https://test.atlassian.net",
            username="test@example.com",
            api_token="test-token",
            jql="project = TEST",
            limit=1,
        )

        mock_response_data = {
            "issues": [mock_issue_simple, mock_issue_adf],
            "isLast": True,
        }

        mock_response = _make_mock_response(json_data=mock_response_data)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            documents = await loader.aload()

            assert len(documents) == 1
            assert documents[0].metadata["key"] == "TEST-123"

    @pytest.mark.asyncio
    async def test_aload_rate_limit_retry(self, jira_loader, mock_issue_simple):
        """Test handling rate limits with retry."""
        mock_response_data = {
            "issues": [mock_issue_simple],
            "isLast": True,
        }

        # First call returns 429, second call succeeds
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers.get = MagicMock(return_value="1")

        mock_200 = _make_mock_response(json_data=mock_response_data)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=[mock_429, mock_200])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                documents = await jira_loader.aload()

                assert len(documents) == 1
                assert mock_client.post.call_count == 2
                mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_aload_http_error(self, jira_loader):
        """Test handling HTTP errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock()
            mock_error = MagicMock()
            mock_error.status_code = 401
            mock_error.text = "Unauthorized"

            from httpx import HTTPStatusError, Request, Response

            mock_post.side_effect = HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(spec=Request),
                response=mock_error,
            )

            mock_client.return_value.__aenter__.return_value.post = mock_post

            with pytest.raises(RuntimeError, match="Failed to fetch Jira issues"):
                await jira_loader.aload()

    @pytest.mark.asyncio
    async def test_aload_missing_httpx(self, jira_loader):
        """Test error when httpx is not installed."""
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx required"):
                await jira_loader.aload()


class TestSyncLoad:
    """Test synchronous loading of Jira issues."""

    def test_load(self, jira_loader, mock_issue_simple):
        """Test synchronous load method."""
        mock_response_data = {
            "issues": [mock_issue_simple],
            "isLast": True,
        }

        mock_response = _make_mock_response(json_data=mock_response_data)

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            documents = jira_loader.load()

            assert len(documents) == 1
            assert documents[0].metadata["key"] == "TEST-123"
