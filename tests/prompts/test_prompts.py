"""Tests for prompt templates."""

from __future__ import annotations

import pytest

from synapsekit.prompts.template import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)

# ------------------------------------------------------------------ #
# PromptTemplate
# ------------------------------------------------------------------ #


class TestPromptTemplate:
    def test_basic_format(self):
        pt = PromptTemplate("Hello, {name}!")
        assert pt.format(name="Alice") == "Hello, Alice!"

    def test_multiple_variables(self):
        pt = PromptTemplate("{greeting}, {name}. You are {age}.")
        result = pt.format(greeting="Hi", name="Bob", age=30)
        assert result == "Hi, Bob. You are 30."

    def test_no_variables(self):
        pt = PromptTemplate("Static prompt.")
        assert pt.format() == "Static prompt."

    def test_missing_variable_raises(self):
        pt = PromptTemplate("{x} and {y}")
        with pytest.raises(KeyError):
            pt.format(x="only")

    def test_returns_string(self):
        pt = PromptTemplate("{val}")
        assert isinstance(pt.format(val="test"), str)


# ------------------------------------------------------------------ #
# ChatPromptTemplate
# ------------------------------------------------------------------ #


class TestChatPromptTemplate:
    def test_format_messages_basic(self):
        cpt = ChatPromptTemplate(
            [
                {"role": "system", "content": "You are {persona}."},
                {"role": "user", "content": "Tell me about {topic}."},
            ]
        )
        messages = cpt.format_messages(persona="a chef", topic="pasta")
        assert messages[0]["role"] == "system"
        assert "a chef" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "pasta" in messages[1]["content"]

    def test_returns_list_of_dicts(self):
        cpt = ChatPromptTemplate([{"role": "user", "content": "Hi {name}"}])
        result = cpt.format_messages(name="Alice")
        assert isinstance(result, list)
        assert all("role" in m and "content" in m for m in result)

    def test_preserves_message_count(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr {x}"},
            {"role": "assistant", "content": "asst"},
        ]
        cpt = ChatPromptTemplate(messages)
        result = cpt.format_messages(x="val")
        assert len(result) == 3

    def test_no_variables(self):
        cpt = ChatPromptTemplate([{"role": "user", "content": "static"}])
        result = cpt.format_messages()
        assert result[0]["content"] == "static"


# ------------------------------------------------------------------ #
# FewShotPromptTemplate
# ------------------------------------------------------------------ #


class TestFewShotPromptTemplate:
    def test_renders_examples_and_suffix(self):
        fsp = FewShotPromptTemplate(
            examples=[
                {"input": "2+2", "output": "4"},
                {"input": "3+3", "output": "6"},
            ],
            example_template="Input: {input}\nOutput: {output}",
            suffix="Input: {question}\nOutput:",
        )
        result = fsp.format(question="5+5")
        assert "Input: 2+2" in result
        assert "Output: 4" in result
        assert "Input: 5+5" in result

    def test_empty_examples(self):
        fsp = FewShotPromptTemplate(
            examples=[],
            example_template="Input: {input}",
            suffix="Question: {q}",
        )
        result = fsp.format(q="hello")
        assert result == "Question: hello"

    def test_single_example(self):
        fsp = FewShotPromptTemplate(
            examples=[{"word": "cat"}],
            example_template="Word: {word}",
            suffix="Now: {x}",
        )
        result = fsp.format(x="dog")
        assert "Word: cat" in result
        assert "Now: dog" in result

    def test_examples_separated_by_newlines(self):
        fsp = FewShotPromptTemplate(
            examples=[{"a": "1"}, {"a": "2"}],
            example_template="{a}",
            suffix="end",
        )
        result = fsp.format()
        assert "1" in result
        assert "2" in result
