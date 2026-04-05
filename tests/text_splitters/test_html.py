"""Tests for HTMLTextSplitter."""

import pytest

from synapsekit.text_splitters import HTMLTextSplitter


def test_html_splitter_default_params() -> None:
    s = HTMLTextSplitter()
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50


def test_html_splitter_validation() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        HTMLTextSplitter(chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        HTMLTextSplitter(chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        HTMLTextSplitter(chunk_size=20, chunk_overlap=20)


def test_empty_text() -> None:
    assert HTMLTextSplitter().split("") == []


def test_split_block_tags_and_strip_html() -> None:
    s = HTMLTextSplitter(chunk_size=1000)
    html = """
    <article>
      <h1>Title</h1>
      <p>Hello <b>world</b>.</p>
      <div>Second block <a href='#'>link</a></div>
    </article>
    """
    chunks = s.split(html)
    assert len(chunks) == 1
    assert "<" not in chunks[0]
    assert "Title" in chunks[0]
    assert "Hello world." in chunks[0]
    assert "Second block link" in chunks[0]


def test_uses_block_boundaries() -> None:
    s = HTMLTextSplitter(chunk_size=40, chunk_overlap=0)
    html = "<p>One short para.</p><p>Two short para.</p><p>Three short para.</p>"
    chunks = s.split(html)
    assert len(chunks) >= 2
    joined = " ".join(chunks)
    assert "One short para." in joined
    assert "Two short para." in joined
    assert "Three short para." in joined


def test_no_block_tags_falls_back_to_plain_text() -> None:
    s = HTMLTextSplitter(chunk_size=1000)
    html = "inline <b>only</b> text"
    chunks = s.split(html)
    assert chunks == ["inline only text"]


def test_handles_script_style_removal() -> None:
    s = HTMLTextSplitter(chunk_size=1000)
    html = "<div>Hello</div><script>alert('x')</script><style>p{color:red}</style><p>World</p>"
    chunks = s.split(html)
    text = " ".join(chunks)
    assert "Hello" in text
    assert "World" in text
    assert "alert" not in text
    assert "color:red" not in text


def test_hard_split_large_block() -> None:
    s = HTMLTextSplitter(chunk_size=30, chunk_overlap=5)
    html = f"<p>{'x' * 90}</p>"
    chunks = s.split(html)
    assert len(chunks) >= 3
    assert all(len(c) <= 35 for c in chunks)


def test_overlap_applied_between_chunks() -> None:
    s = HTMLTextSplitter(chunk_size=25, chunk_overlap=4)
    html = "<p>abcdefghij klmnopqrst uvwxyz</p><p>1234567890 abcdefghij</p>"
    chunks = s.split(html)
    if len(chunks) > 1:
        assert chunks[1][:4] == chunks[0][-4:]
