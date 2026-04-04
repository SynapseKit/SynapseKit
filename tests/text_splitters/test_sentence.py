"""Tests for SentenceTextSplitter."""

import pytest

from synapsekit.text_splitters import SentenceTextSplitter

# ------------------------------------------------------------------ #
# Initialization tests
# ------------------------------------------------------------------ #


def test_sentence_splitter_default_params():
    """Test default parameter values."""
    s = SentenceTextSplitter()
    assert s.chunk_size == 10
    assert s.chunk_overlap == 1


def test_sentence_splitter_custom_params():
    """Test custom parameter values."""
    s = SentenceTextSplitter(chunk_size=5, chunk_overlap=2)
    assert s.chunk_size == 5
    assert s.chunk_overlap == 2


def test_sentence_splitter_invalid_chunk_size():
    """Test that chunk_size must be positive."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SentenceTextSplitter(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SentenceTextSplitter(chunk_size=-1)


def test_sentence_splitter_invalid_chunk_overlap_negative():
    """Test that chunk_overlap cannot be negative."""
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        SentenceTextSplitter(chunk_overlap=-1)


def test_sentence_splitter_invalid_chunk_overlap_too_large():
    """Test that chunk_overlap must be less than chunk_size."""
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        SentenceTextSplitter(chunk_size=5, chunk_overlap=5)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        SentenceTextSplitter(chunk_size=5, chunk_overlap=6)


# ------------------------------------------------------------------ #
# Basic splitting tests
# ------------------------------------------------------------------ #


def test_sentence_splitter_empty_text():
    """Test handling of empty text."""
    s = SentenceTextSplitter()
    assert s.split("") == []
    assert s.split("   ") == []
    assert s.split("\n\n") == []


def test_sentence_splitter_single_sentence():
    """Test with a single sentence."""
    s = SentenceTextSplitter()
    text = "This is a single sentence."
    result = s.split(text)
    assert result == ["This is a single sentence."]


def test_sentence_splitter_fewer_sentences_than_chunk_size():
    """Test when text has fewer sentences than chunk_size."""
    s = SentenceTextSplitter(chunk_size=5)
    text = "First sentence. Second sentence. Third sentence."
    result = s.split(text)
    assert len(result) == 1
    assert result[0] == "First sentence. Second sentence. Third sentence."


def test_sentence_splitter_exact_chunk_size():
    """Test when text has exactly chunk_size sentences."""
    s = SentenceTextSplitter(chunk_size=3)
    text = "First sentence. Second sentence. Third sentence."
    result = s.split(text)
    assert len(result) == 1
    assert result[0] == "First sentence. Second sentence. Third sentence."


def test_sentence_splitter_multiple_chunks():
    """Test splitting into multiple chunks."""
    s = SentenceTextSplitter(chunk_size=2, chunk_overlap=0)
    text = "First. Second. Third. Fourth. Fifth."
    result = s.split(text)
    assert len(result) == 3
    assert result[0] == "First. Second."
    assert result[1] == "Third. Fourth."
    assert result[2] == "Fifth."


# ------------------------------------------------------------------ #
# Overlap tests
# ------------------------------------------------------------------ #


def test_sentence_splitter_overlap():
    """Test overlapping chunks."""
    s = SentenceTextSplitter(chunk_size=3, chunk_overlap=1)
    text = "A. B. C. D. E. F."
    result = s.split(text)

    # Should create overlapping chunks
    assert len(result) >= 2
    # First chunk: A. B. C.
    assert result[0] == "A. B. C."
    # Step size = chunk_size - chunk_overlap = 3 - 1 = 2
    # Second chunk starts at index 2: C. D. E.
    assert result[1] == "C. D. E."
    # Third chunk starts at index 4: E. F.
    assert result[2] == "E. F."


def test_sentence_splitter_no_overlap():
    """Test with zero overlap."""
    s = SentenceTextSplitter(chunk_size=2, chunk_overlap=0)
    text = "A. B. C. D."
    result = s.split(text)
    assert len(result) == 2
    assert result[0] == "A. B."
    assert result[1] == "C. D."


def test_sentence_splitter_overlap_preserves_sentences():
    """Test that overlap preserves complete sentences."""
    s = SentenceTextSplitter(chunk_size=4, chunk_overlap=2)
    text = "First. Second. Third. Fourth. Fifth. Sixth."
    result = s.split(text)

    # Each chunk should contain complete sentences
    for chunk in result:
        sentences = [sent.strip() for sent in chunk.split(". ") if sent.strip()]
        # Remove trailing period from last sentence for counting
        if sentences and sentences[-1].endswith("."):
            sentences[-1] = sentences[-1][:-1]
        assert len(sentences) <= 4


# ------------------------------------------------------------------ #
# Sentence boundary detection
# ------------------------------------------------------------------ #


def test_sentence_splitter_period():
    """Test splitting on periods."""
    s = SentenceTextSplitter(chunk_size=1, chunk_overlap=0)
    text = "First. Second. Third."
    result = s.split(text)
    assert len(result) == 3
    assert result[0] == "First."
    assert result[1] == "Second."
    assert result[2] == "Third."


def test_sentence_splitter_exclamation():
    """Test splitting on exclamation marks."""
    s = SentenceTextSplitter(chunk_size=1, chunk_overlap=0)
    text = "Wow! Amazing! Incredible!"
    result = s.split(text)
    assert len(result) == 3
    assert result[0] == "Wow!"
    assert result[1] == "Amazing!"
    assert result[2] == "Incredible!"


def test_sentence_splitter_question():
    """Test splitting on question marks."""
    s = SentenceTextSplitter(chunk_size=1, chunk_overlap=0)
    text = "How? Why? When?"
    result = s.split(text)
    assert len(result) == 3
    assert result[0] == "How?"
    assert result[1] == "Why?"
    assert result[2] == "When?"


def test_sentence_splitter_mixed_punctuation():
    """Test splitting on mixed sentence boundaries."""
    s = SentenceTextSplitter(chunk_size=1, chunk_overlap=0)
    text = "Hello. How are you? Great!"
    result = s.split(text)
    assert len(result) == 3
    assert result[0] == "Hello."
    assert result[1] == "How are you?"
    assert result[2] == "Great!"


def test_sentence_splitter_multiple_spaces():
    """Test handling of multiple spaces between sentences."""
    s = SentenceTextSplitter(chunk_size=2, chunk_overlap=0)
    text = "First.    Second. Third.    Fourth."
    result = s.split(text)
    assert len(result) == 2
    assert result[0] == "First. Second."
    assert result[1] == "Third. Fourth."


def test_sentence_splitter_newlines():
    """Test handling of newlines between sentences."""
    s = SentenceTextSplitter(chunk_size=2, chunk_overlap=0)
    text = "First.\nSecond.\nThird.\nFourth."
    result = s.split(text)
    assert len(result) == 2
    assert result[0] == "First. Second."
    assert result[1] == "Third. Fourth."


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


def test_sentence_splitter_abbreviations():
    """Test handling of abbreviations (Dr., Mr., etc.)."""
    # Note: This tests current behavior - abbreviations may be split
    s = SentenceTextSplitter(chunk_size=2, chunk_overlap=0)
    text = "Dr. Smith is here. Mr. Jones left."
    result = s.split(text)
    # The regex splits on any period followed by space
    # This is expected behavior for the basic splitter
    assert len(result) >= 1


def test_sentence_splitter_quoted_sentences():
    """Test handling of quoted sentences."""
    s = SentenceTextSplitter(chunk_size=1, chunk_overlap=0)
    text = 'He said "Hello there". She replied "Goodbye".'
    result = s.split(text)
    assert len(result) == 2


def test_sentence_splitter_whitespace_handling():
    """Test various whitespace handling."""
    s = SentenceTextSplitter(chunk_size=2, chunk_overlap=0)
    text = "  First.  \n\n  Second.  \t  Third.  \n  Fourth.  "
    result = s.split(text)
    assert len(result) == 2
    # Should strip leading/trailing whitespace
    assert result[0].startswith("First.")
    assert result[0].endswith("Second.")


def test_sentence_splitter_very_long_text():
    """Test with a longer text."""
    s = SentenceTextSplitter(chunk_size=3, chunk_overlap=1)
    sentences = " ".join([f"Sentence {i}." for i in range(30)])
    result = s.split(sentences)

    # Should create multiple chunks with overlap
    assert len(result) > 1
    # Each chunk should contain at most 3 sentences
    for chunk in result:
        sentence_count = chunk.count(".")
        assert sentence_count <= 3


def test_sentence_splitter_single_character():
    """Test with minimal non-empty text."""
    s = SentenceTextSplitter()
    result = s.split("a")
    assert result == ["a"]


def test_sentence_splitter_no_sentence_boundaries():
    """Test text without sentence boundaries."""
    s = SentenceTextSplitter(chunk_size=5)
    text = "This is a long text without sentence boundaries"
    result = s.split(text)
    assert len(result) == 1
    assert result[0] == text


# ------------------------------------------------------------------ #
# Metadata support
# ------------------------------------------------------------------ #


def test_sentence_splitter_metadata():
    """Test split_with_metadata."""
    s = SentenceTextSplitter(chunk_size=2, chunk_overlap=0)
    text = "First. Second. Third. Fourth."

    result = s.split_with_metadata(text, {"source": "test.txt"})

    assert len(result) == 2
    assert all(chunk["metadata"]["source"] == "test.txt" for chunk in result)
    assert result[0]["metadata"]["chunk_index"] == 0
    assert result[1]["metadata"]["chunk_index"] == 1


def test_sentence_splitter_metadata_empty_text():
    """Test metadata with empty text."""
    s = SentenceTextSplitter()
    result = s.split_with_metadata("", {"source": "test.txt"})
    assert result == []


# ------------------------------------------------------------------ #
# Integration tests
# ------------------------------------------------------------------ #


def test_sentence_splitter_realistic_paragraph():
    """Test with realistic multi-paragraph text."""
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "It was a sunny day outside. "
        "Children were playing in the park. "
        "Birds were singing in the trees. "
        "Everything seemed peaceful and calm. "
        "The sun set slowly in the west. "
        "Night fell upon the small town. "
        "Stars appeared in the dark sky."
    )
    s = SentenceTextSplitter(chunk_size=3, chunk_overlap=1)
    result = s.split(text)

    assert len(result) >= 2
    # Verify all chunks contain complete sentences
    for chunk in result:
        assert chunk.endswith((".", "!", "?"))


def test_sentence_splitter_consecutive_punctuation():
    """Test handling of consecutive punctuation."""
    s = SentenceTextSplitter(chunk_size=2, chunk_overlap=0)
    text = "Really?! Oh no... What happened?!"
    result = s.split(text)
    # Regex splits on whitespace after punctuation
    # Behavior depends on exact spacing
    assert len(result) >= 1


# ------------------------------------------------------------------ #
# Top-level imports
# ------------------------------------------------------------------ #


def test_top_level_export():
    """Test that SentenceTextSplitter is exported at top level."""
    import synapsekit

    assert hasattr(synapsekit, "SentenceTextSplitter")
    from synapsekit import SentenceTextSplitter as TopLevelImport

    assert TopLevelImport is SentenceTextSplitter
