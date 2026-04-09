"""Tests for JSONSplitter."""

from __future__ import annotations

import json

import pytest

from synapsekit.text_splitters.json_splitter import JSONSplitter

# ── Basics ────────────────────────────────────────────────────────


class TestBasicInputs:
    """Empty, whitespace, and scalar inputs."""

    def test_empty_string(self):
        splitter = JSONSplitter()
        assert splitter.split("") == []

    def test_whitespace_only(self):
        splitter = JSONSplitter()
        assert splitter.split("   \n\t  ") == []

    def test_scalar_string(self):
        splitter = JSONSplitter()
        chunks = splitter.split('"hello world"')
        assert chunks == ['"hello world"']

    def test_scalar_number(self):
        splitter = JSONSplitter()
        chunks = splitter.split("42")
        assert chunks == ["42"]

    def test_scalar_bool(self):
        splitter = JSONSplitter()
        assert splitter.split("true") == ["true"]

    def test_scalar_null(self):
        splitter = JSONSplitter()
        assert splitter.split("null") == ["null"]


# ── Invalid JSON ──────────────────────────────────────────────────


class TestInvalidJSON:
    """Should raise ValueError on malformed input."""

    def test_invalid_json_raises(self):
        splitter = JSONSplitter()
        with pytest.raises(ValueError, match="Input is not valid JSON"):
            splitter.split("{not valid json}")

    def test_trailing_comma_raises(self):
        splitter = JSONSplitter()
        with pytest.raises(ValueError, match="Input is not valid JSON"):
            splitter.split('[1, 2, 3,]')


# ── Constructor validation ────────────────────────────────────────


class TestConstructorValidation:
    """Fail fast on bad config, matching CodeSplitter pattern."""

    def test_zero_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            JSONSplitter(chunk_size=0)

    def test_negative_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            JSONSplitter(chunk_size=-10)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            JSONSplitter(chunk_overlap=-1)

    def test_overlap_equals_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            JSONSplitter(chunk_size=100, chunk_overlap=100)

    def test_overlap_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            JSONSplitter(chunk_size=100, chunk_overlap=200)


# ── Array splitting ───────────────────────────────────────────────


class TestArraySplitting:
    """Arrays: each element is a candidate."""

    def test_small_array_single_chunk(self):
        """An array that fits entirely within chunk_size → one chunk."""
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=1000)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert json.loads(chunks[0]) == data

    def test_array_split_across_chunks(self):
        """Array elements get distributed across multiple chunks."""
        data = [{"id": i, "name": f"User {i}"} for i in range(10)]
        text = json.dumps(data)
        # Use a small chunk_size to force multiple chunks
        splitter = JSONSplitter(chunk_size=80)
        chunks = splitter.split(text)
        assert len(chunks) > 1
        # Every chunk must be valid JSON
        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, list)
        # All original elements must be present across all chunks
        all_elements = []
        for chunk in chunks:
            all_elements.extend(json.loads(chunk))
        assert all_elements == data

    def test_empty_array(self):
        splitter = JSONSplitter()
        chunks = splitter.split("[]")
        assert chunks == []

    def test_single_element_array(self):
        data = [{"key": "value"}]
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=1000)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert json.loads(chunks[0]) == data


# ── Object splitting ─────────────────────────────────────────────


class TestObjectSplitting:
    """Objects: each top-level key-value pair is a candidate."""

    def test_small_object_single_chunk(self):
        data = {"name": "Alice", "age": 30, "city": "Paris"}
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=1000)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert json.loads(chunks[0]) == data

    def test_object_split_across_chunks(self):
        """Large object gets split by top-level keys."""
        data = {f"key_{i}": f"value {'x' * 30}" for i in range(10)}
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=100)
        chunks = splitter.split(text)
        assert len(chunks) > 1
        # Every chunk must be a valid JSON object
        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, dict)
        # All keys must be present across chunks
        all_keys = set()
        for chunk in chunks:
            all_keys.update(json.loads(chunk).keys())
        assert all_keys == set(data.keys())

    def test_empty_object(self):
        splitter = JSONSplitter()
        chunks = splitter.split("{}")
        assert chunks == []


# ── Oversized elements ────────────────────────────────────────────


class TestOversizedElements:
    """Single element larger than chunk_size triggers hard split."""

    def test_oversized_array_element(self):
        # One element that is much larger than chunk_size
        big_item = {"data": "x" * 500}
        data = [big_item]
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=100)
        chunks = splitter.split(text)
        assert len(chunks) > 1
        # Hard-split chunks won't be valid JSON, but their concatenation
        # should reconstruct the original serialized candidate
        reconstructed = "".join(chunks)
        assert "x" * 500 in reconstructed

    def test_oversized_object_value(self):
        data = {"big_key": "y" * 500}
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=100)
        chunks = splitter.split(text)
        assert len(chunks) > 1


# ── Overlap ───────────────────────────────────────────────────────


class TestOverlap:
    """chunk_overlap prepends tail of previous chunk."""

    def test_overlap_applied(self):
        data = [{"id": i} for i in range(10)]
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=40, chunk_overlap=8)
        chunks = splitter.split(text)
        assert len(chunks) > 1
        # Every chunk must be valid JSON — overlap is at the item level
        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, list)
        # The last element of chunk[0] must appear as the first element of chunk[1]
        first_last = json.loads(chunks[0])[-1]
        second_first = json.loads(chunks[1])[0]
        assert first_last == second_first, (
            "Overlap must repeat items from the end of the previous chunk, "
            "not raw character tails that produce invalid JSON."
        )

    def test_overlap_chunks_are_valid_json(self):
        """All chunks with overlap must parse as valid JSON arrays."""
        data = [{"id": i, "val": f"x{i}"} for i in range(20)]
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split(text)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            try:
                json.loads(chunk)
            except json.JSONDecodeError as e:
                pytest.fail(f"Chunk {i} is not valid JSON: {e}\nChunk content: {chunk!r}")

    def test_no_overlap(self):
        data = [{"id": i} for i in range(10)]
        text = json.dumps(data)
        splitter_no = JSONSplitter(chunk_size=40, chunk_overlap=0)
        chunks_no = splitter_no.split(text)
        for chunk in chunks_no:
            assert chunk[0] == "["


# ── Nested structures ────────────────────────────────────────────


class TestNestedStructures:
    """Deeply nested JSON is preserved within candidates."""

    def test_nested_objects_in_array(self):
        data = [
            {"user": {"name": "Alice", "address": {"city": "Paris"}}},
            {"user": {"name": "Bob", "address": {"city": "London"}}},
        ]
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=1000)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert json.loads(chunks[0]) == data

    def test_nested_arrays_in_object(self):
        data = {"users": [1, 2, 3], "products": [4, 5, 6]}
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=1000)
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert json.loads(chunks[0]) == data


# ── split_with_metadata (inherited from BaseSplitter) ─────────────


class TestSplitWithMetadata:
    """Verify the inherited split_with_metadata works correctly."""

    def test_metadata_attached(self):
        data = [{"id": 1}, {"id": 2}]
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=1000)
        results = splitter.split_with_metadata(text, metadata={"source": "api"})
        assert len(results) == 1
        assert results[0]["metadata"]["source"] == "api"
        assert results[0]["metadata"]["chunk_index"] == 0
        assert results[0]["text"] == json.dumps(data, separators=(",", ":"))

    def test_metadata_indices_sequential(self):
        data = [{"id": i} for i in range(10)]
        text = json.dumps(data)
        splitter = JSONSplitter(chunk_size=40)
        results = splitter.split_with_metadata(text)
        indices = [r["metadata"]["chunk_index"] for r in results]
        assert indices == list(range(len(results)))


# ── Top-level export ──────────────────────────────────────────────


class TestTopLevelExport:
    """JSONSplitter should be importable from the top-level package."""

    def test_import_from_text_splitters(self):
        from synapsekit.text_splitters import JSONSplitter as ImportedSplitter
        assert ImportedSplitter is JSONSplitter

    def test_import_from_synapsekit(self):
        from synapsekit import JSONSplitter as ImportedSplitter
        assert ImportedSplitter is JSONSplitter


# ── ensure_ascii flag ─────────────────────────────────────────────


class TestEnsureAscii:
    """The ensure_ascii flag controls Unicode escaping in output."""

    def test_unicode_preserved_by_default(self):
        data = [{"name": "日本語"}]
        text = json.dumps(data, ensure_ascii=False)
        splitter = JSONSplitter(chunk_size=1000, ensure_ascii=False)
        chunks = splitter.split(text)
        assert "日本語" in chunks[0]

    def test_unicode_escaped_with_flag(self):
        data = [{"name": "日本語"}]
        text = json.dumps(data, ensure_ascii=False)
        splitter = JSONSplitter(chunk_size=1000, ensure_ascii=True)
        chunks = splitter.split(text)
        assert "日本語" not in chunks[0]
        # But it should parse back correctly
        assert json.loads(chunks[0])[0]["name"] == "日本語"
