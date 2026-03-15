from __future__ import annotations

import pytest

from synapsekit.loaders.base import Document
from synapsekit.loaders.markdown import MarkdownLoader


class TestMarkdownLoader:
    def test_file_not_found(self):
        loader = MarkdownLoader("/nonexistent/file.md")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_basic(self, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("# Title\n\nSome content here.")
        loader = MarkdownLoader(str(p))
        docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "# Title" in docs[0].text
        assert "Some content here." in docs[0].text
        assert docs[0].metadata == {"source": str(p)}

    def test_strip_frontmatter(self, tmp_path):
        p = tmp_path / "front.md"
        p.write_text("---\ntitle: Test\nauthor: Me\n---\n# Heading\n\nBody text.")
        loader = MarkdownLoader(str(p), strip_frontmatter=True)
        docs = loader.load()

        assert "title: Test" not in docs[0].text
        assert "# Heading" in docs[0].text
        assert "Body text." in docs[0].text

    def test_keep_frontmatter(self, tmp_path):
        p = tmp_path / "keep.md"
        p.write_text("---\ntitle: Test\n---\n# Heading")
        loader = MarkdownLoader(str(p), strip_frontmatter=False)
        docs = loader.load()

        assert "title: Test" in docs[0].text
        assert "# Heading" in docs[0].text

    def test_no_frontmatter(self, tmp_path):
        p = tmp_path / "plain.md"
        p.write_text("# Just a heading\n\nParagraph.")
        loader = MarkdownLoader(str(p))
        docs = loader.load()

        assert docs[0].text == "# Just a heading\n\nParagraph."
