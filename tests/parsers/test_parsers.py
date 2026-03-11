"""Tests for output parsers."""

from __future__ import annotations

import pytest

from synapsekit.parsers.json_parser import JSONParser
from synapsekit.parsers.list_parser import ListParser
from synapsekit.parsers.pydantic_parser import PydanticParser

# ------------------------------------------------------------------ #
# JSONParser
# ------------------------------------------------------------------ #


class TestJSONParser:
    def test_parses_clean_json_object(self):
        result = JSONParser().parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_clean_json_array(self):
        result = JSONParser().parse("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_extracts_json_from_text(self):
        text = 'Here is the answer: {"name": "Alice", "age": 30}'
        result = JSONParser().parse(text)
        assert result["name"] == "Alice"

    def test_extracts_array_from_text(self):
        text = 'The list is: ["a", "b", "c"]'
        result = JSONParser().parse(text)
        assert result == ["a", "b", "c"]

    def test_raises_on_unparseable_text(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            JSONParser().parse("no json here at all")

    def test_parses_nested_json(self):
        result = JSONParser().parse('{"a": {"b": [1, 2]}}')
        assert result["a"]["b"] == [1, 2]

    def test_strips_whitespace(self):
        result = JSONParser().parse('  {"x": 1}  ')
        assert result["x"] == 1


# ------------------------------------------------------------------ #
# ListParser
# ------------------------------------------------------------------ #


class TestListParser:
    def test_parses_bullet_list(self):
        text = "- item one\n- item two\n- item three"
        result = ListParser().parse(text)
        assert result == ["item one", "item two", "item three"]

    def test_parses_numbered_list(self):
        text = "1. first\n2. second\n3. third"
        result = ListParser().parse(text)
        assert result == ["first", "second", "third"]

    def test_parses_asterisk_bullets(self):
        text = "* alpha\n* beta"
        result = ListParser().parse(text)
        assert result == ["alpha", "beta"]

    def test_skips_empty_lines(self):
        text = "- a\n\n- b"
        result = ListParser().parse(text)
        assert result == ["a", "b"]

    def test_plain_text_lines(self):
        text = "line one\nline two"
        result = ListParser().parse(text)
        assert result == ["line one", "line two"]

    def test_empty_input(self):
        result = ListParser().parse("")
        assert result == []

    def test_parenthesis_numbered(self):
        text = "1) alpha\n2) beta"
        result = ListParser().parse(text)
        assert result == ["alpha", "beta"]


# ------------------------------------------------------------------ #
# PydanticParser
# ------------------------------------------------------------------ #


class TestPydanticParser:
    def test_parses_into_model(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        parser = PydanticParser(Person)
        result = parser.parse('{"name": "Bob", "age": 25}')
        assert result.name == "Bob"
        assert result.age == 25

    def test_raises_on_missing_field(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class Strict(BaseModel):
            required_field: str

        parser = PydanticParser(Strict)
        with pytest.raises(Exception):  # ValidationError or TypeError
            parser.parse('{"other": "value"}')

    def test_raises_on_bad_json(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        class M(BaseModel):
            x: int

        parser = PydanticParser(M)
        with pytest.raises(ValueError):
            parser.parse("not json")
