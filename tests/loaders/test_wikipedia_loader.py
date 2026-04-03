from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def _make_page(
    title="Test",
    exists=True,
    summary="Summary",
    sections=None,
    categories=None,
    url="https://en.wikipedia.org/wiki/Test",
):
    page = MagicMock()
    page.exists.return_value = exists
    page.summary = summary
    page.sections = sections or []
    page.categories = categories or {}
    page.fullurl = url
    page.title = title
    return page


def _make_mock_api(page):
    wiki = MagicMock()
    wiki.page.return_value = page
    mock_api = MagicMock()
    mock_api.Wikipedia.return_value = wiki
    return mock_api, wiki


def test_import_error_without_wikipedia():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    with patch.dict("sys.modules", {"wikipediaapi": None}):
        loader = WikipediaLoader("Test")
        with pytest.raises(ImportError, match="wikipedia-api"):
            loader.load()


def test_load_returns_document_with_metadata():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    section = MagicMock()
    section.title = "History"
    section.text = "Section text here."
    section.sections = []

    page = _make_page(
        title="Python (programming language)",
        summary="Python is a language.",
        sections=[section],
        categories={"Category:Programming languages": MagicMock()},
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
    )
    mock_api, wiki = _make_mock_api(page)

    with patch.dict("sys.modules", {"wikipediaapi": mock_api}):
        docs = WikipediaLoader("Python (programming language)").load()

    assert len(docs) == 1
    doc = docs[0]
    assert "Python is a language." in doc.text
    assert "Section text here." in doc.text
    assert doc.metadata["title"] == "Python (programming language)"
    assert doc.metadata["url"] == "https://en.wikipedia.org/wiki/Python_(programming_language)"
    assert "Category:Programming languages" in doc.metadata["categories"]
    assert doc.metadata["language"] == "en"
    wiki.page.assert_called_once_with("Python (programming language)")


def test_nonexistent_page_skipped():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    page = _make_page(exists=False)
    mock_api, _ = _make_mock_api(page)

    with patch.dict("sys.modules", {"wikipediaapi": mock_api}):
        docs = WikipediaLoader("NonExistentPageXYZ123").load()

    assert docs == []


def test_pipe_separated_multiple_titles():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    page1 = _make_page(title="Python", summary="Python summary.")
    page2 = _make_page(title="Rust", summary="Rust summary.")

    wiki = MagicMock()
    wiki.page.side_effect = [page1, page2]
    mock_api = MagicMock()
    mock_api.Wikipedia.return_value = wiki

    with patch.dict("sys.modules", {"wikipediaapi": mock_api}):
        docs = WikipediaLoader("Python|Rust", max_results=2).load()

    assert len(docs) == 2
    assert docs[0].metadata["title"] == "Python"
    assert docs[1].metadata["title"] == "Rust"


def test_respects_language_param():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    page = _make_page(title="Python")
    mock_api, _ = _make_mock_api(page)

    with patch.dict("sys.modules", {"wikipediaapi": mock_api}):
        WikipediaLoader("Python", language="de").load()

    call_kwargs = mock_api.Wikipedia.call_args.kwargs
    assert call_kwargs.get("language") == "de"


def test_user_agent_is_set():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    page = _make_page()
    mock_api, _ = _make_mock_api(page)

    with patch.dict("sys.modules", {"wikipediaapi": mock_api}):
        WikipediaLoader("Test").load()

    call_kwargs = mock_api.Wikipedia.call_args.kwargs
    assert "user_agent" in call_kwargs


def test_aload_delegates_to_load():
    from synapsekit.loaders.wikipedia import WikipediaLoader

    loader = WikipediaLoader("Test")
    expected = [MagicMock()]
    with patch.object(loader, "load", return_value=expected) as mock_load:
        result = asyncio.run(loader.aload())

    mock_load.assert_called_once()
    assert result == expected
