from __future__ import annotations

from synapsekit.text_splitters.html import HTMLTextSplitter


class TestHTMLTextSplitter:
    def test_split_empty(self):
        s = HTMLTextSplitter()
        assert s.split("") == []
        assert s.split("   ") == []

    def test_split_small_text(self):
        s = HTMLTextSplitter(chunk_size=200)
        html = "<p>Hello world</p>"
        result = s.split(html)
        assert result == ["Hello world"]

    def test_strips_tags(self):
        s = HTMLTextSplitter(chunk_size=500)
        html = "<div><p>Some <b>bold</b> and <em>italic</em> text</p></div>"
        result = s.split(html)
        assert result == ["Some bold and italic text"]

    def test_splits_on_block_tags(self):
        html = "<h1>Title</h1><p>First paragraph.</p><p>Second paragraph.</p>"
        s = HTMLTextSplitter(chunk_size=30, chunk_overlap=0)
        chunks = s.split(html)
        assert len(chunks) >= 2
        assert any("Title" in c for c in chunks)
        assert any("First" in c for c in chunks)

    def test_handles_nested_html(self):
        html = """
        <section>
            <h2>Section A</h2>
            <p>Content A here.</p>
        </section>
        <section>
            <h2>Section B</h2>
            <p>Content B here.</p>
        </section>
        """
        s = HTMLTextSplitter(chunk_size=40, chunk_overlap=0)
        chunks = s.split(html)
        assert len(chunks) >= 2

    def test_plain_text_passthrough(self):
        s = HTMLTextSplitter(chunk_size=500)
        text = "No HTML tags here at all"
        result = s.split(text)
        assert result == [text]

    def test_large_section_falls_back(self):
        long_text = "word " * 200
        html = f"<div>{long_text}</div>"
        s = HTMLTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = s.split(html)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 100 + 10  # small tolerance

    def test_overlap_applied(self):
        html = "<p>First chunk content here.</p><p>Second chunk content here.</p>"
        s = HTMLTextSplitter(chunk_size=30, chunk_overlap=5)
        chunks = s.split(html)
        if len(chunks) >= 2:
            # second chunk should start with tail of first
            tail = chunks[0][-5:]
            assert chunks[1].startswith(tail)

    def test_list_items(self):
        html = "<ul><li>Item one</li><li>Item two</li><li>Item three</li></ul>"
        s = HTMLTextSplitter(chunk_size=20, chunk_overlap=0)
        chunks = s.split(html)
        assert len(chunks) >= 2

    def test_blockquote_and_pre(self):
        html = "<blockquote>A quote here</blockquote><pre>some code</pre>"
        s = HTMLTextSplitter(chunk_size=500, chunk_overlap=0)
        chunks = s.split(html)
        assert any("quote" in c for c in chunks)
        assert any("code" in c for c in chunks)

    def test_mixed_inline_and_block(self):
        html = "Intro text <p>Paragraph one</p> <span>inline</span> <div>Block two</div>"
        s = HTMLTextSplitter(chunk_size=40, chunk_overlap=0)
        chunks = s.split(html)
        assert len(chunks) >= 1
        full = " ".join(chunks)
        assert "Intro" in full
        assert "Paragraph" in full
