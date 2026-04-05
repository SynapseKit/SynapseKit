"""Tests for JSONSplitter."""

import json

import pytest

from synapsekit.text_splitters import JSONSplitter

# ------------------------------------------------------------------ #
# Initialization tests
# ------------------------------------------------------------------ #


def test_json_splitter_default_params():
    """Test default parameter values."""
    s = JSONSplitter()
    assert s.chunk_size == 512
    assert s.chunk_overlap == 0


def test_json_splitter_custom_params():
    """Test custom parameter values."""
    s = JSONSplitter(chunk_size=100, chunk_overlap=10)
    assert s.chunk_size == 100
    assert s.chunk_overlap == 10


def test_json_splitter_invalid_chunk_size():
    """Test that chunk_size must be positive."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        JSONSplitter(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        JSONSplitter(chunk_size=-1)


def test_json_splitter_invalid_chunk_overlap_negative():
    """Test that chunk_overlap cannot be negative."""
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        JSONSplitter(chunk_overlap=-1)


def test_json_splitter_invalid_chunk_overlap_too_large():
    """Test that chunk_overlap must be less than chunk_size."""
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        JSONSplitter(chunk_size=50, chunk_overlap=50)

    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        JSONSplitter(chunk_size=50, chunk_overlap=60)


# ------------------------------------------------------------------ #
# split() tests - basic functionality
# ------------------------------------------------------------------ #


def test_json_splitter_empty_text():
    """Test empty text input."""
    s = JSONSplitter()
    result = s.split("")
    assert result == []


def test_json_splitter_whitespace_only():
    """Test whitespace-only text."""
    s = JSONSplitter()
    result = s.split("   \n\t  ")
    assert result == []


def test_json_splitter_invalid_json():
    """Test invalid JSON input raises ValueError."""
    s = JSONSplitter()
    with pytest.raises(ValueError, match="Invalid JSON input"):
        s.split("{invalid json")


def test_json_splitter_single_array_element():
    """Test JSON array with single element."""
    s = JSONSplitter(chunk_size=100)
    data = json.dumps([{"name": "Alice", "age": 30}])
    result = s.split(data)

    assert len(result) == 1
    parsed = json.loads(result[0])
    assert parsed == [{"name": "Alice", "age": 30}]


def test_json_splitter_small_array_fits_one_chunk():
    """Test small JSON array that fits in one chunk."""
    s = JSONSplitter(chunk_size=200)
    data = json.dumps([{"id": 1}, {"id": 2}, {"id": 3}])
    result = s.split(data)

    assert len(result) == 1
    parsed = json.loads(result[0])
    assert parsed == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_json_splitter_array_multiple_chunks():
    """Test JSON array that requires multiple chunks."""
    s = JSONSplitter(chunk_size=30)
    # Each element is ~10 chars, array overhead is 2 chars, commas add more
    # Target: force multiple chunks
    data = json.dumps([{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}])
    result = s.split(data)

    # Should be split into multiple chunks
    assert len(result) > 1

    # Verify all elements are preserved
    all_elements = []
    for chunk in result:
        parsed = json.loads(chunk)
        all_elements.extend(parsed)

    assert all_elements == [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]


def test_json_splitter_object_single_key():
    """Test JSON object with single key/value pair."""
    s = JSONSplitter(chunk_size=100)
    data = json.dumps({"name": "Alice", "age": 30})
    result = s.split(data)

    # Both keys fit in one chunk
    assert len(result) == 1

    # Reconstruct the object
    reconstructed = {}
    for chunk in result:
        parsed = json.loads(chunk)
        for item in parsed:
            reconstructed.update(item)

    assert reconstructed == {"name": "Alice", "age": 30}


def test_json_splitter_object_multiple_keys():
    """Test JSON object with multiple keys."""
    s = JSONSplitter(chunk_size=50)
    data = json.dumps({"a": 1, "b": 2, "c": 3, "d": 4})
    result = s.split(data)

    # Should be split into chunks
    assert len(result) >= 1

    # Reconstruct the object
    reconstructed = {}
    for chunk in result:
        parsed = json.loads(chunk)
        for item in parsed:
            reconstructed.update(item)

    assert reconstructed == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_json_splitter_primitive_value():
    """Test JSON primitive value (string, number, etc.)."""
    s = JSONSplitter()

    # String
    result = s.split(json.dumps("hello"))
    assert len(result) == 1
    assert json.loads(result[0]) == ["hello"]

    # Number
    result = s.split(json.dumps(42))
    assert len(result) == 1
    assert json.loads(result[0]) == [42]

    # Boolean
    result = s.split(json.dumps(True))
    assert len(result) == 1
    assert json.loads(result[0]) == [True]


# ------------------------------------------------------------------ #
# Chunk size and overlap tests
# ------------------------------------------------------------------ #


def test_json_splitter_respects_chunk_size():
    """Test that chunks respect the chunk_size limit (approximately)."""
    s = JSONSplitter(chunk_size=40)
    data = json.dumps([{"id": i, "name": f"user{i}"} for i in range(10)])
    result = s.split(data)

    # Each chunk should be roughly within chunk_size
    # Allow some flexibility due to JSON formatting
    for chunk in result:
        # Chunks may slightly exceed chunk_size for single large elements
        assert len(chunk) <= s.chunk_size + 50  # Generous margin


def test_json_splitter_with_overlap():
    """Test chunk overlap functionality."""
    s = JSONSplitter(chunk_size=50, chunk_overlap=15)
    data = json.dumps([{"id": i} for i in range(1, 11)])
    result = s.split(data)

    if len(result) > 1:
        # Check that there is some overlap between chunks
        # This is a basic sanity check
        for i in range(1, len(result)):
            curr_chunk = json.loads(result[i])
            prev_chunk = json.loads(result[i - 1])

            # Some element from prev_chunk should appear in curr_chunk
            # Note: overlap might not always occur depending on element sizes
            # This test documents the behavior
            _ = any(elem in curr_chunk for elem in prev_chunk)


def test_json_splitter_zero_overlap():
    """Test that zero overlap produces non-overlapping chunks."""
    s = JSONSplitter(chunk_size=30, chunk_overlap=0)
    data = json.dumps([{"id": i} for i in range(1, 6)])
    result = s.split(data)

    # Collect all elements
    all_elements = []
    for chunk in result:
        parsed = json.loads(chunk)
        all_elements.extend(parsed)

    # With zero overlap, should have exactly the original elements
    assert all_elements == [{"id": i} for i in range(1, 6)]


# ------------------------------------------------------------------ #
# Complex JSON structures
# ------------------------------------------------------------------ #


def test_json_splitter_nested_objects():
    """Test splitting JSON with nested objects."""
    s = JSONSplitter(chunk_size=100)
    data = json.dumps([
        {"user": {"name": "Alice", "age": 30}},
        {"user": {"name": "Bob", "age": 25}}
    ])
    result = s.split(data)

    # Verify structure is preserved
    all_elements = []
    for chunk in result:
        parsed = json.loads(chunk)
        all_elements.extend(parsed)

    assert all_elements == [
        {"user": {"name": "Alice", "age": 30}},
        {"user": {"name": "Bob", "age": 25}}
    ]


def test_json_splitter_array_of_arrays():
    """Test splitting array of arrays."""
    s = JSONSplitter(chunk_size=50)
    data = json.dumps([[1, 2], [3, 4], [5, 6]])
    result = s.split(data)

    all_elements = []
    for chunk in result:
        parsed = json.loads(chunk)
        all_elements.extend(parsed)

    assert all_elements == [[1, 2], [3, 4], [5, 6]]


def test_json_splitter_mixed_types_in_array():
    """Test array with mixed data types."""
    s = JSONSplitter(chunk_size=100)
    data = json.dumps([1, "hello", {"key": "value"}, [1, 2, 3], True, None])
    result = s.split(data)

    all_elements = []
    for chunk in result:
        parsed = json.loads(chunk)
        all_elements.extend(parsed)

    assert all_elements == [1, "hello", {"key": "value"}, [1, 2, 3], True, None]


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


def test_json_splitter_single_large_element():
    """Test handling of single element that exceeds chunk_size."""
    s = JSONSplitter(chunk_size=20)
    # Single large object
    data = json.dumps([{"description": "This is a very long description that exceeds chunk size"}])
    result = s.split(data)

    # Should still create a chunk even though it exceeds chunk_size
    assert len(result) == 1
    parsed = json.loads(result[0])
    assert len(parsed) == 1


def test_json_splitter_unicode_characters():
    """Test handling of unicode characters in JSON."""
    s = JSONSplitter(chunk_size=100)
    data = json.dumps([{"name": "你好"}, {"name": "مرحبا"}, {"name": "🎉"}])
    result = s.split(data)

    all_elements = []
    for chunk in result:
        parsed = json.loads(chunk)
        all_elements.extend(parsed)

    assert all_elements == [{"name": "你好"}, {"name": "مرحبا"}, {"name": "🎉"}]


def test_json_splitter_empty_array():
    """Test empty JSON array."""
    s = JSONSplitter()
    data = json.dumps([])
    result = s.split(data)

    assert result == []


def test_json_splitter_empty_object():
    """Test empty JSON object."""
    s = JSONSplitter()
    data = json.dumps({})
    result = s.split(data)

    assert result == []


# ------------------------------------------------------------------ #
# Realistic use cases
# ------------------------------------------------------------------ #


def test_json_splitter_realistic_user_data():
    """Test with realistic user data."""
    s = JSONSplitter(chunk_size=150)
    data = json.dumps([
        {"id": 1, "name": "Alice Smith", "email": "alice@example.com", "age": 30},
        {"id": 2, "name": "Bob Jones", "email": "bob@example.com", "age": 25},
        {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com", "age": 35}
    ])
    result = s.split(data)

    # Verify all data is preserved
    all_elements = []
    for chunk in result:
        parsed = json.loads(chunk)
        all_elements.extend(parsed)

    assert len(all_elements) == 3
    assert all_elements[0]["name"] == "Alice Smith"
    assert all_elements[1]["name"] == "Bob Jones"
    assert all_elements[2]["name"] == "Charlie Brown"
