"""Tests for JSONSplitter."""

import json

import pytest

from synapsekit.text_splitters import JSONSplitter


def test_json_splitter_default_params() -> None:
    s = JSONSplitter()
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50


def test_json_splitter_validation() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        JSONSplitter(chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        JSONSplitter(chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        JSONSplitter(chunk_size=20, chunk_overlap=20)


def test_empty_text() -> None:
    assert JSONSplitter().split("") == []


def test_invalid_json() -> None:
    s = JSONSplitter()
    with pytest.raises(ValueError, match="Invalid JSON"):
        s.split("{invalid json}")


def test_split_json_array() -> None:
    s = JSONSplitter(chunk_size=100, chunk_overlap=0)
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ]
    json_text = json.dumps(data)
    chunks = s.split(json_text)

    # Each item should be a separate chunk candidate
    assert len(chunks) >= 1
    # Verify all data is present
    combined = "\n".join(chunks)
    assert "Alice" in combined
    assert "Bob" in combined
    assert "Charlie" in combined


def test_split_json_object() -> None:
    s = JSONSplitter(chunk_size=80, chunk_overlap=0)
    data = {
        "user1": {"name": "Alice", "age": 30},
        "user2": {"name": "Bob", "age": 25},
        "user3": {"name": "Charlie", "age": 35},
    }
    json_text = json.dumps(data)
    chunks = s.split(json_text)

    # Each top-level key should be a split candidate
    assert len(chunks) >= 1
    combined = "\n".join(chunks)
    assert "user1" in combined or "Alice" in combined
    assert "user2" in combined or "Bob" in combined
    assert "user3" in combined or "Charlie" in combined


def test_split_json_primitive() -> None:
    s = JSONSplitter()
    # String
    chunks = s.split('"hello"')
    assert chunks == ['"hello"']

    # Number
    chunks = s.split("42")
    assert chunks == ["42"]

    # Boolean
    chunks = s.split("true")
    assert chunks == ["true"]

    # Null
    chunks = s.split("null")
    assert chunks == ["null"]


def test_large_array_items_merged() -> None:
    s = JSONSplitter(chunk_size=150, chunk_overlap=0)
    data = [{"id": i, "value": f"item_{i}"} for i in range(10)]
    json_text = json.dumps(data)
    chunks = s.split(json_text)

    # Multiple small items should be merged into fewer chunks
    assert len(chunks) < len(data)
    # Verify all data is present
    combined = "\n".join(chunks)
    for i in range(10):
        assert f"item_{i}" in combined


def test_oversized_single_element() -> None:
    s = JSONSplitter(chunk_size=50, chunk_overlap=10)
    # Create a single large object that exceeds chunk_size
    data = [{"key": "x" * 100}]
    json_text = json.dumps(data)
    chunks = s.split(json_text)

    # Should hard-split the oversized element
    assert len(chunks) > 1
    # Verify the oversized content is split
    combined = "".join(chunks)
    assert "x" * 100 in combined


def test_chunk_overlap() -> None:
    s = JSONSplitter(chunk_size=60, chunk_overlap=10)
    data = [
        {"id": 1, "name": "A"},
        {"id": 2, "name": "B"},
        {"id": 3, "name": "C"},
    ]
    json_text = json.dumps(data)
    chunks = s.split(json_text)

    # With overlap, consecutive chunks should share some content
    if len(chunks) > 1:
        # Check that last 10 chars of chunk[i-1] appear in chunk[i]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-10:]
            assert chunks[i].startswith(tail)


def test_unicode_characters() -> None:
    s = JSONSplitter(chunk_size=1000)
    data = {"greeting": "Hello 世界", "emoji": "😀🚀"}
    json_text = json.dumps(data, ensure_ascii=False)
    chunks = s.split(json_text)

    combined = "\n".join(chunks)
    assert "世界" in combined
    assert "😀" in combined


def test_nested_structure() -> None:
    s = JSONSplitter(chunk_size=200, chunk_overlap=0)
    data = {
        "users": [
            {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        ]
    }
    json_text = json.dumps(data)
    chunks = s.split(json_text)

    # Each top-level key is treated as a candidate
    combined = "\n".join(chunks)
    assert "Alice" in combined
    assert "Bob" in combined
    assert "NYC" in combined


def test_empty_array() -> None:
    s = JSONSplitter()
    chunks = s.split("[]")
    assert chunks == []


def test_empty_object() -> None:
    s = JSONSplitter()
    chunks = s.split("{}")
    assert chunks == []


def test_split_with_metadata() -> None:
    s = JSONSplitter(chunk_size=100, chunk_overlap=0)
    data = [{"id": 1}, {"id": 2}]
    json_text = json.dumps(data)
    result = s.split_with_metadata(json_text, metadata={"source": "test.json"})

    assert len(result) >= 1
    for idx, chunk_dict in enumerate(result):
        assert "text" in chunk_dict
        assert "metadata" in chunk_dict
        assert chunk_dict["metadata"]["source"] == "test.json"
        assert chunk_dict["metadata"]["chunk_index"] == idx
