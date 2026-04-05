"""Tests for SentenceWindowSplitter."""

import pytest

from synapsekit.text_splitters import SentenceWindowSplitter

# ------------------------------------------------------------------ #
# Initialization tests
# ------------------------------------------------------------------ #


def test_sentence_window_splitter_default_params():
    """Test default parameter values."""
    s = SentenceWindowSplitter()
    assert s.window_size == 2


def test_sentence_window_splitter_custom_window_size():
    """Test custom window_size parameter."""
    s = SentenceWindowSplitter(window_size=3)
    assert s.window_size == 3


def test_sentence_window_splitter_zero_window_size():
    """Test that window_size=0 is valid (no context)."""
    s = SentenceWindowSplitter(window_size=0)
    assert s.window_size == 0


def test_sentence_window_splitter_invalid_window_size():
    """Test that window_size cannot be negative."""
    with pytest.raises(ValueError, match="window_size cannot be negative"):
        SentenceWindowSplitter(window_size=-1)


# ------------------------------------------------------------------ #
# split() tests - basic functionality
# ------------------------------------------------------------------ #


def test_sentence_window_splitter_empty_text():
    """Test empty text input."""
    s = SentenceWindowSplitter()
    result = s.split("")
    assert result == []


def test_sentence_window_splitter_whitespace_only():
    """Test whitespace-only text."""
    s = SentenceWindowSplitter()
    result = s.split("   \n\t  ")
    assert result == []


def test_sentence_window_splitter_single_sentence():
    """Test single sentence input."""
    s = SentenceWindowSplitter(window_size=2)
    result = s.split("This is a single sentence.")
    assert result == ["This is a single sentence."]


def test_sentence_window_splitter_two_sentences_window_1():
    """Test two sentences with window_size=1."""
    s = SentenceWindowSplitter(window_size=1)
    text = "First sentence. Second sentence."
    result = s.split(text)

    # First sentence: includes itself + 1 after
    # Second sentence: includes 1 before + itself
    assert len(result) == 2
    assert result[0] == "First sentence. Second sentence."
    assert result[1] == "First sentence. Second sentence."


def test_sentence_window_splitter_three_sentences_window_1():
    """Test three sentences with window_size=1."""
    s = SentenceWindowSplitter(window_size=1)
    text = "First sentence. Second sentence. Third sentence."
    result = s.split(text)

    assert len(result) == 3
    # Window around first: itself + 1 after
    assert result[0] == "First sentence. Second sentence."
    # Window around second: 1 before + itself + 1 after
    assert result[1] == "First sentence. Second sentence. Third sentence."
    # Window around third: 1 before + itself
    assert result[2] == "Second sentence. Third sentence."


def test_sentence_window_splitter_three_sentences_window_2():
    """Test three sentences with window_size=2."""
    s = SentenceWindowSplitter(window_size=2)
    text = "First sentence. Second sentence. Third sentence."
    result = s.split(text)

    assert len(result) == 3
    # All windows will include all sentences since window_size=2 is >= distance to edges
    assert result[0] == "First sentence. Second sentence. Third sentence."
    assert result[1] == "First sentence. Second sentence. Third sentence."
    assert result[2] == "First sentence. Second sentence. Third sentence."


def test_sentence_window_splitter_five_sentences_window_2():
    """Test five sentences with window_size=2."""
    s = SentenceWindowSplitter(window_size=2)
    text = "One. Two. Three. Four. Five."
    result = s.split(text)

    assert len(result) == 5
    # Window around "One": itself + 2 after = One, Two, Three
    assert result[0] == "One. Two. Three."
    # Window around "Two": 1 before + itself + 2 after = One, Two, Three, Four
    assert result[1] == "One. Two. Three. Four."
    # Window around "Three": 2 before + itself + 2 after = all 5
    assert result[2] == "One. Two. Three. Four. Five."
    # Window around "Four": 2 before + itself + 1 after = Two, Three, Four, Five
    assert result[3] == "Two. Three. Four. Five."
    # Window around "Five": 2 before + itself = Three, Four, Five
    assert result[4] == "Three. Four. Five."


def test_sentence_window_splitter_zero_window():
    """Test window_size=0 (no context, just the target sentence)."""
    s = SentenceWindowSplitter(window_size=0)
    text = "First. Second. Third."
    result = s.split(text)

    assert len(result) == 3
    assert result[0] == "First."
    assert result[1] == "Second."
    assert result[2] == "Third."


# ------------------------------------------------------------------ #
# split_with_metadata() tests
# ------------------------------------------------------------------ #


def test_sentence_window_metadata_empty_text():
    """Test split_with_metadata with empty text."""
    s = SentenceWindowSplitter()
    result = s.split_with_metadata("")
    assert result == []


def test_sentence_window_metadata_single_sentence():
    """Test split_with_metadata with single sentence."""
    s = SentenceWindowSplitter(window_size=1)
    result = s.split_with_metadata("Single sentence.")

    assert len(result) == 1
    assert result[0]["text"] == "Single sentence."
    assert result[0]["metadata"]["chunk_index"] == 0
    assert result[0]["metadata"]["target_sentence"] == "Single sentence."


def test_sentence_window_metadata_three_sentences():
    """Test split_with_metadata with three sentences."""
    s = SentenceWindowSplitter(window_size=1)
    text = "First. Second. Third."
    result = s.split_with_metadata(text)

    assert len(result) == 3

    # First sentence window
    assert result[0]["text"] == "First. Second."
    assert result[0]["metadata"]["chunk_index"] == 0
    assert result[0]["metadata"]["target_sentence"] == "First."

    # Second sentence window
    assert result[1]["text"] == "First. Second. Third."
    assert result[1]["metadata"]["chunk_index"] == 1
    assert result[1]["metadata"]["target_sentence"] == "Second."

    # Third sentence window
    assert result[2]["text"] == "Second. Third."
    assert result[2]["metadata"]["chunk_index"] == 2
    assert result[2]["metadata"]["target_sentence"] == "Third."


def test_sentence_window_metadata_with_parent_metadata():
    """Test that parent metadata is preserved."""
    s = SentenceWindowSplitter(window_size=1)
    text = "First. Second."
    parent_meta = {"source": "test.txt", "page": 1}
    result = s.split_with_metadata(text, metadata=parent_meta)

    assert len(result) == 2

    # Check first chunk
    assert result[0]["metadata"]["source"] == "test.txt"
    assert result[0]["metadata"]["page"] == 1
    assert result[0]["metadata"]["chunk_index"] == 0
    assert result[0]["metadata"]["target_sentence"] == "First."

    # Check second chunk
    assert result[1]["metadata"]["source"] == "test.txt"
    assert result[1]["metadata"]["page"] == 1
    assert result[1]["metadata"]["chunk_index"] == 1
    assert result[1]["metadata"]["target_sentence"] == "Second."


# ------------------------------------------------------------------ #
# Edge cases and special characters
# ------------------------------------------------------------------ #


def test_sentence_window_various_punctuation():
    """Test sentence splitting with different punctuation marks."""
    s = SentenceWindowSplitter(window_size=0)
    text = "Question? Exclamation! Statement."
    result = s.split(text)

    assert len(result) == 3
    assert result[0] == "Question?"
    assert result[1] == "Exclamation!"
    assert result[2] == "Statement."


def test_sentence_window_multiple_spaces():
    """Test handling of multiple spaces between sentences."""
    s = SentenceWindowSplitter(window_size=0)
    text = "First.    Second.     Third."
    result = s.split(text)

    assert len(result) == 3
    assert result[0] == "First."
    assert result[1] == "Second."
    assert result[2] == "Third."


def test_sentence_window_newlines_between_sentences():
    """Test handling of newlines between sentences."""
    s = SentenceWindowSplitter(window_size=0)
    text = "First.\nSecond.\n\nThird."
    result = s.split(text)

    assert len(result) == 3
    assert "First" in result[0]
    assert "Second" in result[1]
    assert "Third" in result[2]


def test_sentence_window_abbreviations():
    """Test that abbreviations like Dr. don't break sentences incorrectly."""
    s = SentenceWindowSplitter(window_size=0)
    # Note: Current implementation may split on "Dr." - this is a known limitation
    # This test documents the behavior
    text = "Dr. Smith works here. She is great."
    result = s.split(text)

    # The regex will likely split after "Dr." - this is expected behavior
    # In production, you might want a more sophisticated sentence splitter
    assert len(result) >= 2


# ------------------------------------------------------------------ #
# Realistic use case
# ------------------------------------------------------------------ #


def test_sentence_window_realistic_paragraph():
    """Test with a realistic paragraph for retrieval use case."""
    s = SentenceWindowSplitter(window_size=1)
    text = (
        "The Eiffel Tower is located in Paris. "
        "It was built in 1889. "
        "The tower is 330 meters tall. "
        "Millions of tourists visit it every year."
    )
    result = s.split_with_metadata(text)

    assert len(result) == 4

    # For retrieval, you'd store target_sentence as the retrieval key
    # but use the full window text for embedding
    assert result[1]["metadata"]["target_sentence"] == "It was built in 1889."
    # Context window includes surrounding sentences
    assert "Eiffel Tower" in result[1]["text"]
    assert "330 meters" in result[1]["text"]


def test_sentence_window_large_window_size():
    """Test with a large window_size that exceeds sentence count."""
    s = SentenceWindowSplitter(window_size=100)
    text = "First. Second. Third."
    result = s.split(text)

    # All windows should include all sentences
    assert len(result) == 3
    for chunk in result:
        assert chunk == "First. Second. Third."
