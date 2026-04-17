from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.loaders import Document
from synapsekit.loaders.sitemap import SitemapLoader, _parse_sitemap


# ---------------------------------------------------------------------------
# Sitemap XML fixtures
# ---------------------------------------------------------------------------

SIMPLE_SITEMAP = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/page1</loc>
    <lastmod>2024-01-01</lastmod>
  </url>
  <url>
    <loc>https://example.com/page2</loc>
  </url>
</urlset>
"""

SITEMAP_INDEX = """\
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://example.com/sitemap-posts.xml</loc>
  </sitemap>
  <sitemap>
    <loc>https://example.com/sitemap-pages.xml</loc>
  </sitemap>
</sitemapindex>
"""

CHILD_SITEMAP = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/post/1</loc>
    <lastmod>2024-03-01</lastmod>
  </url>
</urlset>
"""


# ---------------------------------------------------------------------------
# Initialisation validation
# ---------------------------------------------------------------------------


def test_init_requires_url() -> None:
    with pytest.raises(ValueError, match="url must be provided"):
        SitemapLoader(url="")


def test_init_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        SitemapLoader(url="ftp://example.com/sitemap.xml")


def test_init_defaults() -> None:
    loader = SitemapLoader(url="https://example.com/sitemap.xml")
    assert loader._limit is None
    assert loader._filter_urls is None


# ---------------------------------------------------------------------------
# _parse_sitemap unit tests (pure, no HTTP)
# ---------------------------------------------------------------------------


def test_parse_sitemap_url_entries() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    entries, index_urls = _parse_sitemap(SIMPLE_SITEMAP, "https://example.com")

    assert len(entries) == 2
    assert entries[0]["loc"] == "https://example.com/page1"
    assert entries[0]["lastmod"] == "2024-01-01"
    assert entries[1]["loc"] == "https://example.com/page2"
    assert entries[1]["lastmod"] == ""
    assert index_urls == []


def test_parse_sitemap_index() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    entries, index_urls = _parse_sitemap(SITEMAP_INDEX, "https://example.com")

    assert entries == []
    assert "https://example.com/sitemap-posts.xml" in index_urls
    assert "https://example.com/sitemap-pages.xml" in index_urls


# ---------------------------------------------------------------------------
# Missing dependencies
# ---------------------------------------------------------------------------


def test_load_import_error_missing_httpx() -> None:
    with patch.dict("sys.modules", {"httpx": None}):
        loader = SitemapLoader(url="https://example.com/sitemap.xml")
        with pytest.raises(ImportError, match="httpx required"):
            loader.load()


# ---------------------------------------------------------------------------
# Normal load (mocked HTTP)
# ---------------------------------------------------------------------------


def _make_response(text: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    return resp


def _make_async_client(responses: dict[str, str]) -> MagicMock:
    """Build a mock AsyncClient whose .get() returns different bodies per URL."""

    async def fake_get(url: str, **kwargs: object) -> MagicMock:
        body = responses.get(url, "")
        return _make_response(body)

    client = MagicMock()
    client.get = fake_get
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


PAGE_HTML = "<html><body><p>Hello from page one</p></body></html>"
PAGE2_HTML = "<html><body><p>Content of page two</p></body></html>"


def test_load_returns_documents() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    responses = {
        "https://example.com/sitemap.xml": SIMPLE_SITEMAP,
        "https://example.com/page1": PAGE_HTML,
        "https://example.com/page2": PAGE2_HTML,
    }
    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = _make_async_client(responses)

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        loader = SitemapLoader(url="https://example.com/sitemap.xml")
        docs = loader.load()

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)


def test_load_metadata_correctness() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    responses = {
        "https://example.com/sitemap.xml": SIMPLE_SITEMAP,
        "https://example.com/page1": PAGE_HTML,
        "https://example.com/page2": PAGE2_HTML,
    }
    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = _make_async_client(responses)

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        loader = SitemapLoader(url="https://example.com/sitemap.xml")
        docs = loader.load()

    page1 = next(d for d in docs if d.metadata["url"] == "https://example.com/page1")
    assert page1.metadata["source"] == "sitemap"
    assert page1.metadata["lastmod"] == "2024-01-01"


def test_load_respects_limit() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    responses = {
        "https://example.com/sitemap.xml": SIMPLE_SITEMAP,
        "https://example.com/page1": PAGE_HTML,
    }
    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = _make_async_client(responses)

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        loader = SitemapLoader(url="https://example.com/sitemap.xml", limit=1)
        docs = loader.load()

    assert len(docs) == 1


def test_load_filter_urls() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    responses = {
        "https://example.com/sitemap.xml": SIMPLE_SITEMAP,
        "https://example.com/page1": PAGE_HTML,
    }
    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = _make_async_client(responses)

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        loader = SitemapLoader(
            url="https://example.com/sitemap.xml",
            filter_urls=["page1"],
        )
        docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["url"] == "https://example.com/page1"


def test_load_empty_sitemap() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    empty_sitemap = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
</urlset>
"""
    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = _make_async_client(
        {"https://example.com/sitemap.xml": empty_sitemap}
    )

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        loader = SitemapLoader(url="https://example.com/sitemap.xml")
        docs = loader.load()

    assert docs == []


def test_load_sitemap_index_follows_children() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    responses = {
        "https://example.com/sitemap.xml": SITEMAP_INDEX,
        "https://example.com/sitemap-posts.xml": CHILD_SITEMAP,
        "https://example.com/sitemap-pages.xml": (
            '<?xml version="1.0"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            "<url><loc>https://example.com/about</loc></url>"
            "</urlset>"
        ),
        "https://example.com/post/1": "<html><body><p>Post one</p></body></html>",
        "https://example.com/about": "<html><body><p>About page</p></body></html>",
    }
    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = _make_async_client(responses)

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        loader = SitemapLoader(url="https://example.com/sitemap.xml")
        docs = loader.load()

    urls = {d.metadata["url"] for d in docs}
    assert "https://example.com/post/1" in urls
    assert "https://example.com/about" in urls


def test_load_skips_failed_pages() -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")

    # page2 raises an exception — should be silently skipped
    async def fake_get(url: str, **kwargs: object) -> MagicMock:
        if url == "https://example.com/sitemap.xml":
            return _make_response(SIMPLE_SITEMAP)
        if url == "https://example.com/page1":
            return _make_response(PAGE_HTML)
        raise ConnectionError("network error")

    client = MagicMock()
    client.get = fake_get
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = client

    with patch.dict("sys.modules", {"httpx": mock_httpx}):
        loader = SitemapLoader(url="https://example.com/sitemap.xml")
        docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["url"] == "https://example.com/page1"
