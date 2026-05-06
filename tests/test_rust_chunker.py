"""Tests for the Rust extension module — skipped if not built.

These tests validate that the Rust implementations produce identical output
to the pure-Python implementations.
"""

from __future__ import annotations

import pytest

try:
    from synapsekit._rust_core import (
        character_split,
        deserialize_metadata_list,
        fast_cache_key,
        recursive_split,
        serialize_metadata_list,
    )

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust extension not built")


# ── chunker tests ────────────────────────────────────────────────────────


class TestRecursiveSplit:
    def test_empty(self):
        assert recursive_split("", 100, 0, ["\n\n"]) == []

    def test_whitespace_only(self):
        assert recursive_split("   \n  ", 100, 0, ["\n\n"]) == []

    def test_short_text(self):
        assert recursive_split("hello", 100, 0, ["\n\n"]) == ["hello"]

    def test_paragraph_split(self):
        text = "para one\n\npara two\n\npara three"
        chunks = recursive_split(text, 20, 0, ["\n\n", "\n", " "])
        assert len(chunks) >= 2
        joined = "".join(chunks)
        assert "para one" in joined
        assert "para three" in joined

    def test_hard_split(self):
        text = "a" * 100
        chunks = recursive_split(text, 30, 0, ["\n\n"])
        assert all(len(c) <= 30 for c in chunks)

    def test_overlap(self):
        text = "aaaa\n\nbbbb\n\ncccc"
        chunks = recursive_split(text, 6, 2, ["\n\n"])
        assert len(chunks) >= 2
        # Second chunk should start with tail of first
        assert chunks[1].startswith(chunks[0][-2:])

    def test_recursive_sub_splitting(self):
        """Oversized parts should be recursively split by next separator."""
        text = "sentence one. sentence two. sentence three\n\nshort"
        seps = ["\n\n", ". ", " "]
        chunks = recursive_split(text, 20, 0, seps)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk) <= 20, f"chunk too long: {chunk!r}"

    def test_matches_python(self):
        """Rust output must match Python output exactly."""
        from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter

        text = "The quick brown fox jumps over the lazy dog. " * 10
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        # Temporarily disable Rust to get Python output
        import synapsekit.text_splitters.recursive as mod

        old = mod._rust_split
        mod._rust_split = None
        try:
            py_result = splitter.split(text)
        finally:
            mod._rust_split = old

        rust_result = recursive_split(
            text, 50, 10, ["\n\n", "\n", ". ", " "]
        )
        assert rust_result == py_result


class TestCharacterSplit:
    def test_empty(self):
        assert character_split("", "\n", 100, 0) == []

    def test_short_text(self):
        assert character_split("hello", "\n", 100, 0) == ["hello"]

    def test_newline_split(self):
        text = "line1\nline2\nline3"
        chunks = character_split(text, "\n", 10, 0)
        assert len(chunks) >= 2

    def test_no_separator_found(self):
        text = "a" * 100
        chunks = character_split(text, "\n", 30, 0)
        assert all(len(c) <= 30 for c in chunks)

    def test_matches_python(self):
        """Rust output must match Python output exactly."""
        from synapsekit.text_splitters.character import CharacterTextSplitter

        text = "line one\nline two\nline three\nline four\nline five"
        splitter = CharacterTextSplitter(separator="\n", chunk_size=20, chunk_overlap=5)
        import synapsekit.text_splitters.character as mod

        old = mod._rust_split
        mod._rust_split = None
        try:
            py_result = splitter.split(text)
        finally:
            mod._rust_split = old

        rust_result = character_split(text, "\n", 20, 5)
        assert rust_result == py_result


# ── cache key tests ──────────────────────────────────────────────────────


class TestFastCacheKey:
    def test_deterministic(self):
        k1 = fast_cache_key("gpt-4", "hello", 0.7, 100)
        k2 = fast_cache_key("gpt-4", "hello", 0.7, 100)
        assert k1 == k2

    def test_different_inputs(self):
        k1 = fast_cache_key("gpt-4", "hello", 0.7, 100)
        k2 = fast_cache_key("gpt-4", "world", 0.7, 100)
        assert k1 != k2

    def test_different_model(self):
        k1 = fast_cache_key("gpt-4", "hello", 0.7, 100)
        k2 = fast_cache_key("gpt-3.5", "hello", 0.7, 100)
        assert k1 != k2

    def test_list_input(self):
        messages = [{"role": "user", "content": "hi"}]
        k = fast_cache_key("gpt-4", messages, 0.7, 100)
        assert isinstance(k, str)
        assert len(k) == 32  # xxh3_128 hex = 32 chars

    def test_matches_python_xxhash(self):
        """Rust cache key must match Python xxhash path."""
        pytest.importorskip("xxhash")
        import synapsekit.llm._cache as mod
        from synapsekit.llm._cache import AsyncLRUCache

        # Force Python path
        old_rust = mod._rust_cache_key
        mod._rust_cache_key = None
        try:
            py_key = AsyncLRUCache.make_key("gpt-4", "hello world", 0.7, 100)
        finally:
            mod._rust_cache_key = old_rust

        rust_key = fast_cache_key("gpt-4", "hello world", 0.7, 100)
        assert rust_key == py_key


# ── JSON serialization tests ─────────────────────────────────────────────


class TestSerializeMetadata:
    def test_basic(self):
        data = [{"key": "value", "num": 42}]
        result = serialize_metadata_list(data)
        assert len(result) == 1
        assert '"key"' in result[0]
        assert "42" in result[0]

    def test_empty_list(self):
        assert serialize_metadata_list([]) == []

    def test_nested(self):
        data = [{"a": {"b": [1, 2, 3]}, "c": None}]
        result = serialize_metadata_list(data)
        assert len(result) == 1
        import json
        parsed = json.loads(result[0])
        assert parsed["a"]["b"] == [1, 2, 3]
        assert parsed["c"] is None

    def test_roundtrip(self):
        original = [
            {"name": "test", "score": 0.95, "tags": ["a", "b"]},
            {"empty": {}, "flag": True},
        ]
        serialized = serialize_metadata_list(original)
        deserialized = deserialize_metadata_list(serialized)
        assert len(deserialized) == 2
        assert deserialized[0]["name"] == "test"
        assert deserialized[0]["score"] == 0.95
        assert deserialized[0]["tags"] == ["a", "b"]
        assert deserialized[1]["flag"] is True

    def test_unicode(self):
        data = [{"emoji": "\U0001f600", "cjk": "\u4e16\u754c"}]
        result = serialize_metadata_list(data)
        deserialized = deserialize_metadata_list(result)
        assert deserialized[0]["emoji"] == "\U0001f600"
        assert deserialized[0]["cjk"] == "\u4e16\u754c"


class TestDeserializeMetadata:
    def test_basic(self):
        data = ['{"a": 1}', '{"b": "hello"}']
        result = deserialize_metadata_list(data)
        assert len(result) == 2
        assert result[0]["a"] == 1
        assert result[1]["b"] == "hello"

    def test_empty(self):
        assert deserialize_metadata_list([]) == []

    def test_invalid_json_raises(self):
        with pytest.raises((ValueError, TypeError)):
            deserialize_metadata_list(["not json"])
