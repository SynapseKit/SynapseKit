"""Tests for MarkdownTextSplitter."""

from __future__ import annotations

from synapsekit.text_splitters.markdown import MarkdownTextSplitter


class TestMarkdownTextSplitter:
    """Tests for MarkdownTextSplitter."""

    def test_split_empty(self):
        s = MarkdownTextSplitter()
        assert s.split("") == []
        assert s.split("   ") == []

    def test_split_small_text(self):
        s = MarkdownTextSplitter(chunk_size=200)
        text = "Just a small paragraph."
        assert s.split(text) == [text]

    def test_split_by_headers(self):
        text = "## Section A\nContent for section A.\n## Section B\nContent for section B."
        s = MarkdownTextSplitter(chunk_size=60, chunk_overlap=0)
        chunks = s.split(text)
        assert len(chunks) >= 2
        assert any("Section A" in c for c in chunks)
        assert any("Section B" in c for c in chunks)

    def test_preserves_header_context(self):
        text = (
            "# Title\n"
            "## Section A\n"
            "Content A here that is within chunk size.\n"
            "## Section B\n"
            "Content B here."
        )
        s = MarkdownTextSplitter(chunk_size=80, chunk_overlap=0)
        chunks = s.split(text)
        assert len(chunks) >= 2
        # Each chunk about a section should carry the parent # Title
        for chunk in chunks:
            if "Section A" in chunk or "Section B" in chunk:
                assert "# Title" in chunk

    def test_nested_headers(self):
        text = "# Top\n## Mid\n### Deep\nDeep content here."
        s = MarkdownTextSplitter(chunk_size=80, chunk_overlap=0)
        chunks = s.split(text)
        # The deep content chunk should preserve the full hierarchy
        for chunk in chunks:
            if "Deep content" in chunk:
                assert "# Top" in chunk
                assert "## Mid" in chunk
                assert "### Deep" in chunk

    def test_falls_back_for_oversized(self):
        # A large block of text with no headers should be split via fallback
        text = "word " * 500  # 2500 chars
        s = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = s.split(text)
        assert len(chunks) >= 2

    def test_horizontal_rule_split(self):
        # Horizontal rules (---) are used as a fallback separator
        text = (
            "First section content that is long enough.\n"
            "---\n"
            "Second section content that is also long enough."
        )
        s = MarkdownTextSplitter(chunk_size=60, chunk_overlap=0)
        chunks = s.split(text)
        assert len(chunks) >= 2

    def test_custom_headers_to_split_on(self):
        text = "## Only H2\nContent under H2.\n### Only H3\nContent under H3."
        # Only split on ## headers
        s = MarkdownTextSplitter(
            chunk_size=60,
            chunk_overlap=0,
            headers_to_split_on=[("##", "Header2")],
        )
        chunks = s.split(text)
        assert len(chunks) >= 1
        # ### should NOT cause a split since it's not configured
        # so "### Only H3" stays with its preceding section
        found = [c for c in chunks if "Only H3" in c]
        assert len(found) >= 1

    def test_chunk_overlap(self):
        text = (
            "## Section A\n"
            "Content for section A is here.\n"
            "## Section B\n"
            "Content for section B is here."
        )
        s = MarkdownTextSplitter(chunk_size=60, chunk_overlap=5)
        chunks = s.split(text)
        if len(chunks) >= 2:
            # Second chunk should start with the tail of the first
            tail = chunks[0][-5:]
            assert chunks[1].startswith(tail)

    def test_no_headers(self):
        # Text without any headers should fall back to recursive splitting
        text = "First paragraph of text.\n\nSecond paragraph of text.\n\nThird paragraph of text."
        s = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)
        chunks = s.split(text)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) for c in chunks)
