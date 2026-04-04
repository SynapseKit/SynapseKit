from unittest.mock import MagicMock, patch

from synapsekit.loaders import RSSLoader


@patch("feedparser.parse")
def test_rss_loader(mock_parse):
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
    mock_parse.return_value = mock_feed

    loader = RSSLoader("https://example.com/feed.xml")
    docs = loader.load()

    mock_parse.assert_called_once_with("https://example.com/feed.xml")

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
