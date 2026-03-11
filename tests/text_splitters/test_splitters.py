"""Tests for the text_splitters package."""

from synapsekit.text_splitters import (
    BaseSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenAwareSplitter,
)

# ------------------------------------------------------------------ #
# BaseSplitter ABC
# ------------------------------------------------------------------ #


def test_base_splitter_is_abstract():
    import pytest

    with pytest.raises(TypeError):
        BaseSplitter()  # type: ignore[abstract]


# ------------------------------------------------------------------ #
# CharacterTextSplitter
# ------------------------------------------------------------------ #


def test_character_splitter_empty():
    s = CharacterTextSplitter()
    assert s.split("") == []
    assert s.split("   ") == []


def test_character_splitter_small_text():
    s = CharacterTextSplitter(chunk_size=100)
    assert s.split("Hello world") == ["Hello world"]


def test_character_splitter_splits_on_separator():
    text = "paragraph one\n\nparagraph two\n\nparagraph three"
    s = CharacterTextSplitter(separator="\n\n", chunk_size=20, chunk_overlap=0)
    chunks = s.split(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 20 for c in chunks)


def test_character_splitter_overlap():
    text = "aaaa\n\nbbbb\n\ncccc"
    s = CharacterTextSplitter(separator="\n\n", chunk_size=6, chunk_overlap=2)
    chunks = s.split(text)
    assert len(chunks) >= 2
    # Overlap means second chunk should start with tail of first
    assert chunks[1].startswith(chunks[0][-2:])


# ------------------------------------------------------------------ #
# RecursiveCharacterTextSplitter
# ------------------------------------------------------------------ #


def test_recursive_splitter_empty():
    s = RecursiveCharacterTextSplitter()
    assert s.split("") == []


def test_recursive_splitter_small():
    s = RecursiveCharacterTextSplitter(chunk_size=100)
    assert s.split("short") == ["short"]


def test_recursive_splitter_paragraphs():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    s = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=0)
    chunks = s.split(text)
    assert len(chunks) >= 2


def test_recursive_splitter_sentences():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    s = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0)
    chunks = s.split(text)
    assert len(chunks) >= 2


def test_recursive_splitter_hard_split():
    text = "a" * 200
    s = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
    chunks = s.split(text)
    assert len(chunks) >= 4
    assert all(len(c) <= 50 for c in chunks)


def test_recursive_splitter_custom_separators():
    text = "a|b|c|d|e"
    s = RecursiveCharacterTextSplitter(chunk_size=3, chunk_overlap=0, separators=["|"])
    chunks = s.split(text)
    assert len(chunks) >= 3


def test_recursive_splitter_overlap():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    s = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=5)
    chunks = s.split(text)
    if len(chunks) >= 2:
        assert chunks[1].startswith(chunks[0][-5:])


# ------------------------------------------------------------------ #
# TokenAwareSplitter
# ------------------------------------------------------------------ #


def test_token_splitter_empty():
    s = TokenAwareSplitter(max_tokens=100)
    assert s.split("") == []


def test_token_splitter_small():
    s = TokenAwareSplitter(max_tokens=100)
    assert s.split("short") == ["short"]


def test_token_splitter_respects_token_budget():
    # 50 tokens * 4 chars/token = 200 chars max
    text = "word " * 100  # 500 chars
    s = TokenAwareSplitter(max_tokens=50, chunk_overlap=0)
    chunks = s.split(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 200 for c in chunks)


def test_token_splitter_custom_chars_per_token():
    text = "a " * 100  # 200 chars
    s = TokenAwareSplitter(max_tokens=10, chunk_overlap=0, chars_per_token=2)
    chunks = s.split(text)
    # 10 tokens * 2 chars = 20 char budget
    assert len(chunks) >= 5


# ------------------------------------------------------------------ #
# Backward compatibility
# ------------------------------------------------------------------ #


def test_pipeline_text_splitter_alias():
    from synapsekit.rag.pipeline import TextSplitter

    assert TextSplitter is RecursiveCharacterTextSplitter


def test_text_splitter_compat_api():
    """TextSplitter alias should work exactly like old code."""
    from synapsekit.rag.pipeline import TextSplitter

    s = TextSplitter(chunk_size=100, chunk_overlap=10)
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = s.split(text)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)


# ------------------------------------------------------------------ #
# Top-level imports
# ------------------------------------------------------------------ #


def test_top_level_exports():
    import synapsekit

    assert hasattr(synapsekit, "BaseSplitter")
    assert hasattr(synapsekit, "CharacterTextSplitter")
    assert hasattr(synapsekit, "RecursiveCharacterTextSplitter")
    assert hasattr(synapsekit, "TokenAwareSplitter")
    assert hasattr(synapsekit, "SemanticSplitter")
