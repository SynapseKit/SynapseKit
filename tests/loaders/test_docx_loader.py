from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.docx import DocxLoader


class TestDocxLoader:
    def test_file_not_found(self):
        loader = DocxLoader("/nonexistent/file.docx")
        with pytest.raises(FileNotFoundError):
            loader.load()

    @patch("synapsekit.loaders.docx.os.path.exists", return_value=True)
    def test_import_error(self, _mock_exists):
        loader = DocxLoader("test.docx")
        with patch.dict("sys.modules", {"docx": None}):
            with pytest.raises(ImportError, match="python-docx"):
                loader.load()

    @patch("synapsekit.loaders.docx.os.path.exists", return_value=True)
    def test_load_paragraphs(self, _mock_exists):
        mock_doc = MagicMock()
        mock_doc.paragraphs = [
            MagicMock(text="Hello world"),
            MagicMock(text=""),
            MagicMock(text="Second paragraph"),
        ]

        mock_docx_module = types.ModuleType("docx")
        mock_docx_cls = MagicMock(return_value=mock_doc)
        mock_docx_module.Document = mock_docx_cls

        with patch.dict(sys.modules, {"docx": mock_docx_module}):
            loader = DocxLoader("test.docx")
            docs = loader.load()

            mock_docx_cls.assert_called_once_with("test.docx")
            assert len(docs) == 1
            assert isinstance(docs[0], Document)
            assert docs[0].text == "Hello world\nSecond paragraph"
            assert docs[0].metadata == {"source": "test.docx"}
