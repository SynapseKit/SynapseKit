"""Tests for the _json fast-serialization wrapper."""

from __future__ import annotations

from synapsekit._json import dumps, dumps_bytes, loads


class TestDumps:
    def test_simple_dict(self):
        result = dumps({"a": 1, "b": "hello"})
        assert isinstance(result, str)
        parsed = loads(result)
        assert parsed == {"a": 1, "b": "hello"}

    def test_nested_structure(self):
        obj = {"list": [1, 2, 3], "nested": {"x": True, "y": None}}
        result = dumps(obj)
        assert loads(result) == obj

    def test_empty_dict(self):
        assert loads(dumps({})) == {}

    def test_empty_list(self):
        assert loads(dumps([])) == []

    def test_unicode(self):
        obj = {"emoji": "\U0001f600", "cjk": "\u4e16\u754c"}
        assert loads(dumps(obj)) == obj

    def test_numeric_types(self):
        obj = {"int": 42, "float": 3.14, "neg": -1, "zero": 0}
        assert loads(dumps(obj)) == obj


class TestDumpsBytes:
    def test_returns_bytes(self):
        result = dumps_bytes({"key": "value"})
        assert isinstance(result, bytes)

    def test_roundtrip(self):
        obj = {"a": [1, 2, 3]}
        assert loads(dumps_bytes(obj)) == obj


class TestLoads:
    def test_from_string(self):
        assert loads('{"a": 1}') == {"a": 1}

    def test_from_bytes(self):
        assert loads(b'{"a": 1}') == {"a": 1}

    def test_array(self):
        assert loads("[1, 2, 3]") == [1, 2, 3]

    def test_invalid_json_raises(self):
        import pytest

        with pytest.raises((ValueError, TypeError)):
            loads("not json")
