from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.base import Document


def test_import_error_without_arxiv():
    from synapsekit.loaders.arxiv import ArXivLoader

    with patch.dict("sys.modules", {"arxiv": None}):
        loader = ArXivLoader("2101.00001")
        with pytest.raises(ImportError, match="arxiv"):
            loader.load()


def test_uses_id_list_for_arxiv_id():
    from synapsekit.loaders.arxiv import ArXivLoader

    search_mock = MagicMock()
    search_mock.results.return_value = []

    mock_arxiv = MagicMock()
    mock_arxiv.Search = MagicMock(return_value=search_mock)

    with patch.dict("sys.modules", {"arxiv": mock_arxiv}):
        ArXivLoader("2101.00001").load()

    mock_arxiv.Search.assert_called_once()
    kwargs = mock_arxiv.Search.call_args.kwargs
    assert kwargs.get("id_list") == ["2101.00001"]


def test_load_returns_document_with_metadata():
    from synapsekit.loaders.arxiv import ArXivLoader

    result = MagicMock()
    result.title = "Test Paper"
    result.summary = "Abstract"
    result.published = datetime(2024, 1, 2)
    result.entry_id = "http://arxiv.org/abs/2101.00001"
    result.pdf_url = "http://arxiv.org/pdf/2101.00001"
    result.get_short_id.return_value = "2101.00001"
    author = MagicMock()
    author.name = "Author One"
    result.authors = [author]
    result.download_pdf.return_value = "fake.pdf"

    search_mock = MagicMock()
    search_mock.results.return_value = [result]

    mock_arxiv = MagicMock()
    mock_arxiv.Search = MagicMock(return_value=search_mock)

    with patch.dict("sys.modules", {"arxiv": mock_arxiv}):
        with patch("synapsekit.loaders.arxiv.PDFLoader.load") as mock_pdf_load:
            mock_pdf_load.return_value = [
                Document(text="Page 1", metadata={}),
                Document(text="Page 2", metadata={}),
            ]
            docs = ArXivLoader("some query").load()

    assert len(docs) == 1
    doc = docs[0]
    assert "Page 1" in doc.text
    assert doc.metadata["title"] == "Test Paper"
    assert doc.metadata["abstract"] == "Abstract"
    assert doc.metadata["authors"] == ["Author One"]
    assert doc.metadata["published"] == "2024-01-02T00:00:00"
    assert doc.metadata["arxiv_id"] == "2101.00001"
