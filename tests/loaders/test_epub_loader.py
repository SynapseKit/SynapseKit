from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.epub import EPUBLoader, _html_to_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(name: str, content: bytes, item_type: int = 9) -> MagicMock:
    """Build a fake ebooklib item (ITEM_DOCUMENT = 9)."""
    item = MagicMock()
    item.get_name.return_value = name
    item.get_content.return_value = content
    item.get_type.return_value = item_type
    return item


def _fake_ebooklib(
    items: list[MagicMock], title: str = "My Book", author: str = "Author"
) -> dict[str, ModuleType]:
    """Inject a minimal fake ebooklib into sys.modules."""
    item_document = 9

    book = MagicMock()
    book.get_metadata.side_effect = lambda ns, key: (
        [(title, {})] if key == "title" else [(author, {})] if key == "creator" else []
    )
    book.get_items.return_value = items

    epub_mod = ModuleType("ebooklib.epub")
    epub_mod.ITEM_DOCUMENT = item_document  # type: ignore[attr-defined]
    epub_mod.read_epub = MagicMock(return_value=book)  # type: ignore[attr-defined]

    pkg = ModuleType("ebooklib")
    pkg.epub = epub_mod  # type: ignore[attr-defined]

    return {"ebooklib": pkg, "ebooklib.epub": epub_mod}


# ---------------------------------------------------------------------------
# _html_to_text unit tests (no mocking needed)
# ---------------------------------------------------------------------------


class TestHtmlToText:
    def test_strips_tags(self):
        assert _html_to_text(b"<p>Hello <b>World</b></p>") == "Hello World"

    def test_skips_script(self):
        assert "alert" not in _html_to_text(b"<script>alert(1)</script>Text")

    def test_skips_style(self):
        assert "color" not in _html_to_text(b"<style>body{color:red}</style>Text")

    def test_normalises_whitespace(self):
        result = _html_to_text(b"<p>  foo   bar  </p>")
        assert "  " not in result
        assert "foo bar" in result

    def test_empty_html(self):
        assert _html_to_text(b"") == ""


# ---------------------------------------------------------------------------
# EPUBLoader tests
# ---------------------------------------------------------------------------


class TestEPUBLoader:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            EPUBLoader("/nonexistent/file.epub").load()

    def test_missing_dependency_raises(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        with patch.dict("sys.modules", {"ebooklib": None, "ebooklib.epub": None}):
            with pytest.raises(ImportError, match="ebooklib required"):
                EPUBLoader(str(p)).load()

    def test_returns_list_of_documents(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        items = [_make_item("ch1.xhtml", b"<p>Chapter one</p>")]
        with patch.dict("sys.modules", _fake_ebooklib(items)):
            docs = EPUBLoader(str(p)).load()
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_one_document_per_chapter(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        items = [
            _make_item("ch1.xhtml", b"<p>Chapter one</p>"),
            _make_item("ch2.xhtml", b"<p>Chapter two</p>"),
        ]
        with patch.dict("sys.modules", _fake_ebooklib(items)):
            docs = EPUBLoader(str(p)).load()
        assert len(docs) == 2

    def test_metadata_title_and_author(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        items = [_make_item("ch1.xhtml", b"<p>Content</p>")]
        with patch.dict("sys.modules", _fake_ebooklib(items, title="Dune", author="Herbert")):
            docs = EPUBLoader(str(p)).load()
        assert docs[0].metadata["title"] == "Dune"
        assert docs[0].metadata["author"] == "Herbert"

    def test_metadata_source_and_chapter(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        items = [_make_item("ch1.xhtml", b"<p>Content</p>")]
        with patch.dict("sys.modules", _fake_ebooklib(items)):
            docs = EPUBLoader(str(p)).load()
        assert docs[0].metadata["source"] == str(p)
        assert docs[0].metadata["chapter"] == "ch1.xhtml"

    def test_empty_chapters_are_skipped(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        items = [
            _make_item("ch1.xhtml", b"<p>Real content</p>"),
            _make_item("empty.xhtml", b"   "),
        ]
        with patch.dict("sys.modules", _fake_ebooklib(items)):
            docs = EPUBLoader(str(p)).load()
        assert len(docs) == 1

    def test_non_document_items_are_skipped(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        items = [
            _make_item("ch1.xhtml", b"<p>Real content</p>", item_type=9),
            _make_item("style.css", b"body{}", item_type=3),
        ]
        with patch.dict("sys.modules", _fake_ebooklib(items)):
            docs = EPUBLoader(str(p)).load()
        assert len(docs) == 1

    def test_missing_metadata_defaults_to_empty_string(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        items = [_make_item("ch1.xhtml", b"<p>Content</p>")]
        fake = _fake_ebooklib(items)
        book = fake["ebooklib.epub"].read_epub.return_value
        book.get_metadata.side_effect = lambda ns, key: []
        with patch.dict("sys.modules", fake):
            docs = EPUBLoader(str(p)).load()
        assert docs[0].metadata["title"] == ""
        assert docs[0].metadata["author"] == ""

    def test_empty_epub_returns_empty_list(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        with patch.dict("sys.modules", _fake_ebooklib([])):
            docs = EPUBLoader(str(p)).load()
        assert docs == []

    def test_text_content_extracted(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        items = [_make_item("ch1.xhtml", b"<p>Hello World</p>")]
        with patch.dict("sys.modules", _fake_ebooklib(items)):
            docs = EPUBLoader(str(p)).load()
        assert "Hello World" in docs[0].text

    async def test_aload(self, tmp_path):
        p = tmp_path / "book.epub"
        p.write_bytes(b"")
        items = [_make_item("ch1.xhtml", b"<p>Async chapter</p>")]
        with patch.dict("sys.modules", _fake_ebooklib(items)):
            docs = await EPUBLoader(str(p)).aload()
        assert len(docs) == 1
        assert "Async chapter" in docs[0].text
