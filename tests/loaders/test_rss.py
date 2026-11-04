from unittest.mock import MagicMock, patch

import pytest


def test_rss_loader():
    from synapsekit.loaders.rss import RSSLoader

    mock_feedparser = MagicMock()
    mock_feed = MagicMock()
    mock_feed.entries = [
        {
            "title": "Entry 1",
            "summary": "Summary 1",
            "content": [{"value": "Content 1"}],
            "published": "2024-01-01T00:00:00Z",
            "link": "https://example.com/1",
            "author": "Author 1",
        },
        {
            "title": "Entry 2",
            "summary": "Summary 2",
            # No content to test fallback to summary
            "published": "2024-01-02T00:00:00Z",
            "link": "https://example.com/2",
            "author": "Author 2",
        },
        {
            "title": "Entry 3",
            # No content or summary
        },
    ]
    mock_feedparser.parse.return_value = mock_feed

    with patch.dict("sys.modules", {"feedparser": mock_feedparser}):
        loader = RSSLoader("https://example.com/feed.xml")
        docs = loader.load()

    mock_feedparser.parse.assert_called_once_with("https://example.com/feed.xml")

    assert len(docs) == 3

    assert docs[0].text == "Content 1"
    assert docs[0].metadata["title"] == "Entry 1"
    assert docs[0].metadata["published"] == "2024-01-01T00:00:00Z"
    assert docs[0].metadata["link"] == "https://example.com/1"
    assert docs[0].metadata["author"] == "Author 1"

    assert docs[1].text == "Summary 2"
    assert docs[1].metadata["title"] == "Entry 2"
    assert docs[1].metadata["published"] == "2024-01-02T00:00:00Z"
    assert docs[1].metadata["link"] == "https://example.com/2"
    assert docs[1].metadata["author"] == "Author 2"

    assert docs[2].text == ""
    assert docs[2].metadata["title"] == "Entry 3"
    assert "published" not in docs[2].metadata
    assert "link" not in docs[2].metadata
    assert "author" not in docs[2].metadata


def test_import_error_without_feedparser():
    from synapsekit.loaders.rss import RSSLoader

    with patch.dict("sys.modules", {"feedparser": None}):
        loader = RSSLoader("https://example.com/feed.xml")
        with pytest.raises(ImportError, match="feedparser required"):
            loader.load()


def test_validate_allows_windows_drive_path():
    from synapsekit.loaders.rss import RSSLoader

    # Backslash and forward-slash drive-letter paths must be treated as local
    # files, not rejected as an unknown URL scheme ("c").
    RSSLoader(r"C:\feeds\news.xml")._validate()
    RSSLoader("C:/feeds/news.xml")._validate()
    RSSLoader("d:/data/feed.rss")._validate()


def test_validate_allows_bare_local_path_and_inline_xml():
    from synapsekit.loaders.rss import RSSLoader

    RSSLoader("./feeds/news.xml")._validate()
    RSSLoader("<rss><channel></channel></rss>")._validate()


def test_validate_still_rejects_non_http_schemes():
    from synapsekit.loaders.rss import RSSLoader

    # The drive-letter allowance must not reopen the file:// / ftp:// SSRF holes.
    for url in ("file:///etc/passwd", "ftp://example.com/feed.xml"):
        with pytest.raises(ValueError):
            RSSLoader(url)._validate()
