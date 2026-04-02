from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_import_error_without_wikipedia():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    with patch.dict("sys.modules", {"wikipediaapi": None}):
        loader = WikipediaLoader("Test")
        with pytest.raises(ImportError, match="wikipedia-api"):
            loader.load()


def test_search_query_uses_results():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    page = MagicMock()
    page.exists.return_value = True
    page.summary = "Summary"
    page.sections = []
    page.categories = {}
    page.fullurl = "https://example.com"
    page.title = "Title"

    wiki = MagicMock()
    wiki.search.return_value = ["Title"]
    wiki.page.return_value = page

    mock_api = MagicMock()
    mock_api.Wikipedia = MagicMock(return_value=wiki)
    mock_api.ExtractFormat = MagicMock()

    with patch.dict("sys.modules", {"wikipediaapi": mock_api}):
        docs = WikipediaLoader("query", max_results=1).load()

    assert len(docs) == 1
    wiki.search.assert_called_once()
    wiki.page.assert_called_once_with("Title")


def test_load_returns_document_with_metadata():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    section = MagicMock()
    section.title = "Section"
    section.text = "Section text"
    section.sections = []

    page = MagicMock()
    page.exists.return_value = True
    page.summary = "Summary"
    page.sections = [section]
    page.categories = {"Category:A": MagicMock()}
    page.fullurl = "https://example.com/wiki/Test"
    page.title = "Test"

    wiki = MagicMock()
    wiki.search = None
    wiki.page.return_value = page

    mock_api = MagicMock()
    mock_api.Wikipedia = MagicMock(return_value=wiki)
    mock_api.ExtractFormat = MagicMock()

    with patch.dict("sys.modules", {"wikipediaapi": mock_api}):
        docs = WikipediaLoader("Test", language="en").load()

    assert len(docs) == 1
    doc = docs[0]
    assert "Summary" in doc.text
    assert "Section text" in doc.text
    assert doc.metadata["title"] == "Test"
    assert doc.metadata["url"] == "https://example.com/wiki/Test"
    assert doc.metadata["categories"] == ["Category:A"]
    assert doc.metadata["language"] == "en"
