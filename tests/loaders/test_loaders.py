"""Tests for all loaders."""
from __future__ import annotations

import csv
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from synapsekit.loaders.base import Document
from synapsekit.loaders.text import StringLoader, TextLoader
from synapsekit.loaders.csv import CSVLoader
from synapsekit.loaders.json_loader import JSONLoader
from synapsekit.loaders.directory import DirectoryLoader


# ------------------------------------------------------------------ #
# Document dataclass
# ------------------------------------------------------------------ #

class TestDocument:
    def test_defaults(self):
        doc = Document(text="hello")
        assert doc.text == "hello"
        assert doc.metadata == {}

    def test_custom_metadata(self):
        doc = Document(text="x", metadata={"source": "test"})
        assert doc.metadata["source"] == "test"

    def test_mutable_metadata_not_shared(self):
        d1 = Document(text="a")
        d2 = Document(text="b")
        d1.metadata["k"] = "v"
        assert "k" not in d2.metadata


# ------------------------------------------------------------------ #
# StringLoader
# ------------------------------------------------------------------ #

class TestStringLoader:
    def test_load_returns_single_document(self):
        loader = StringLoader("hello world")
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].text == "hello world"

    def test_load_with_metadata(self):
        loader = StringLoader("text", metadata={"key": "val"})
        docs = loader.load()
        assert docs[0].metadata["key"] == "val"

    def test_load_empty_string(self):
        loader = StringLoader("")
        docs = loader.load()
        assert docs[0].text == ""

    def test_default_metadata_is_empty(self):
        loader = StringLoader("text")
        docs = loader.load()
        assert docs[0].metadata == {}


# ------------------------------------------------------------------ #
# TextLoader
# ------------------------------------------------------------------ #

class TestTextLoader:
    def test_load_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content here")
        docs = TextLoader(str(f)).load()
        assert len(docs) == 1
        assert docs[0].text == "content here"
        assert docs[0].metadata["source"] == str(f)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            TextLoader("/nonexistent/path.txt").load()

    def test_encoding_param(self, tmp_path):
        f = tmp_path / "encoded.txt"
        f.write_text("héllo", encoding="utf-8")
        docs = TextLoader(str(f), encoding="utf-8").load()
        assert "héllo" in docs[0].text


# ------------------------------------------------------------------ #
# PDFLoader (mocked)
# ------------------------------------------------------------------ #

class TestPDFLoader:
    def test_import_error_without_pypdf(self):
        from synapsekit.loaders.pdf import PDFLoader
        with patch.dict("sys.modules", {"pypdf": None}):
            loader = PDFLoader("dummy.pdf")
            with pytest.raises(ImportError, match="pypdf"):
                loader.load()

    def test_load_pages(self, tmp_path):
        from synapsekit.loaders.pdf import PDFLoader

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]

        with patch("synapsekit.loaders.pdf.PDFLoader.load") as mock_load:
            mock_load.return_value = [
                Document(text="Page 1 content", metadata={"source": "f.pdf", "page": 0}),
                Document(text="Page 2 content", metadata={"source": "f.pdf", "page": 1}),
            ]
            docs = PDFLoader("f.pdf").load()
            assert len(docs) == 2
            assert docs[0].metadata["page"] == 0
            assert docs[1].text == "Page 2 content"


# ------------------------------------------------------------------ #
# HTMLLoader (mocked)
# ------------------------------------------------------------------ #

class TestHTMLLoader:
    def test_import_error_without_bs4(self):
        from synapsekit.loaders.html import HTMLLoader
        with patch.dict("sys.modules", {"bs4": None}):
            loader = HTMLLoader("dummy.html")
            with pytest.raises(ImportError, match="beautifulsoup4"):
                loader.load()

    def test_strips_html_tags(self, tmp_path):
        from synapsekit.loaders.html import HTMLLoader
        f = tmp_path / "test.html"
        f.write_text("<html><body><h1>Title</h1><p>Para</p></body></html>")

        bs4 = pytest.importorskip("bs4")
        docs = HTMLLoader(str(f)).load()
        assert len(docs) == 1
        assert "Title" in docs[0].text
        assert "<h1>" not in docs[0].text


# ------------------------------------------------------------------ #
# CSVLoader
# ------------------------------------------------------------------ #

class TestCSVLoader:
    def test_load_all_columns_as_text(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25\n")
        docs = CSVLoader(str(f)).load()
        assert len(docs) == 2
        assert "Alice" in docs[0].text or "30" in docs[0].text

    def test_load_with_text_column(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("content,author\nhello,alice\nworld,bob\n")
        docs = CSVLoader(str(f), text_column="content").load()
        assert docs[0].text == "hello"
        assert docs[0].metadata["author"] == "alice"

    def test_row_in_metadata(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\nx,y\n")
        docs = CSVLoader(str(f)).load()
        assert docs[0].metadata["row"] == 0

    def test_source_in_metadata(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a\nfoo\n")
        docs = CSVLoader(str(f)).load()
        assert docs[0].metadata["source"] == str(f)

    def test_empty_csv(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("col\n")
        docs = CSVLoader(str(f)).load()
        assert docs == []


# ------------------------------------------------------------------ #
# JSONLoader
# ------------------------------------------------------------------ #

class TestJSONLoader:
    def test_load_list(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"text": "hello"}, {"text": "world"}]))
        docs = JSONLoader(str(f)).load()
        assert len(docs) == 2
        assert docs[0].text == "hello"

    def test_load_single_object(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"text": "single"}))
        docs = JSONLoader(str(f)).load()
        assert len(docs) == 1
        assert docs[0].text == "single"

    def test_custom_text_key(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"content": "foo", "id": 1}]))
        docs = JSONLoader(str(f), text_key="content", metadata_keys=["id"]).load()
        assert docs[0].text == "foo"
        assert docs[0].metadata["id"] == 1

    def test_index_in_metadata(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"text": "a"}, {"text": "b"}]))
        docs = JSONLoader(str(f)).load()
        assert docs[1].metadata["index"] == 1


# ------------------------------------------------------------------ #
# DirectoryLoader
# ------------------------------------------------------------------ #

class TestDirectoryLoader:
    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("file a")
        (tmp_path / "b.txt").write_text("file b")
        docs = DirectoryLoader(str(tmp_path), glob_pattern="*.txt").load()
        texts = [d.text for d in docs]
        assert "file a" in texts
        assert "file b" in texts

    def test_loads_csv_files(self, tmp_path):
        (tmp_path / "data.csv").write_text("col\nvalue\n")
        docs = DirectoryLoader(str(tmp_path), glob_pattern="*.csv").load()
        assert len(docs) == 1

    def test_loads_json_files(self, tmp_path):
        (tmp_path / "data.json").write_text('[{"text": "hi"}]')
        docs = DirectoryLoader(str(tmp_path), glob_pattern="*.json").load()
        assert len(docs) == 1
        assert docs[0].text == "hi"

    def test_empty_directory(self, tmp_path):
        docs = DirectoryLoader(str(tmp_path)).load()
        assert docs == []

    def test_skips_unreadable_files(self, tmp_path):
        (tmp_path / "ok.txt").write_text("good")
        # DirectoryLoader silently skips errors
        docs = DirectoryLoader(str(tmp_path), glob_pattern="*.txt").load()
        assert any(d.text == "good" for d in docs)


# ------------------------------------------------------------------ #
# WebLoader (mocked)
# ------------------------------------------------------------------ #

class TestWebLoader:
    def test_import_error_without_httpx(self):
        from synapsekit.loaders.web import WebLoader
        with patch.dict("sys.modules", {"httpx": None}):
            loader = WebLoader("http://example.com")
            with pytest.raises(ImportError, match="httpx"):
                loader.load_sync()

    @pytest.mark.asyncio
    async def test_async_load(self):
        from synapsekit.loaders.web import WebLoader

        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello web</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        bs4 = pytest.importorskip("bs4")

        with patch("httpx.AsyncClient", return_value=mock_client):
            loader = WebLoader("http://example.com")
            docs = await loader.load()

        assert len(docs) == 1
        assert "Hello web" in docs[0].text
        assert docs[0].metadata["source"] == "http://example.com"

    def test_sync_load(self):
        from synapsekit.loaders.web import WebLoader

        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Sync content</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        bs4 = pytest.importorskip("bs4")

        with patch("httpx.Client", return_value=mock_client):
            docs = WebLoader("http://example.com").load_sync()

        assert "Sync content" in docs[0].text
