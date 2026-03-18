"""Tests for v0.6.0 features: 6 built-in tools, 3 LLM providers, 2 retrieval strategies."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

# ------------------------------------------------------------------ #
# Tool imports
# ------------------------------------------------------------------ #


def test_import_new_tools():
    from synapsekit import (
        DateTimeTool,
        FileListTool,
        FileWriteTool,
        HTTPRequestTool,
        JSONQueryTool,
        RegexTool,
    )

    assert DateTimeTool is not None
    assert FileListTool is not None
    assert FileWriteTool is not None
    assert HTTPRequestTool is not None
    assert JSONQueryTool is not None
    assert RegexTool is not None


# ------------------------------------------------------------------ #
# DateTimeTool
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_datetime_now():
    from synapsekit import DateTimeTool

    tool = DateTimeTool()
    r = await tool.run(action="now")
    assert not r.is_error
    assert "T" in r.output  # ISO format


@pytest.mark.asyncio
async def test_datetime_now_utc():
    from synapsekit import DateTimeTool

    tool = DateTimeTool()
    r = await tool.run(action="now", tz="utc")
    assert not r.is_error
    assert "+00:00" in r.output


@pytest.mark.asyncio
async def test_datetime_parse():
    from synapsekit import DateTimeTool

    tool = DateTimeTool()
    r = await tool.run(action="parse", value="2026-03-12T10:30:00")
    assert not r.is_error
    assert "2026-03-12" in r.output


@pytest.mark.asyncio
async def test_datetime_format():
    from synapsekit import DateTimeTool

    tool = DateTimeTool()
    r = await tool.run(action="format", value="2026-03-12T10:30:00", fmt="%B %d, %Y")
    assert r.output == "March 12, 2026"


# ------------------------------------------------------------------ #
# FileWriteTool
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_file_write():
    from synapsekit import FileWriteTool

    tool = FileWriteTool()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        path = f.name

    try:
        r = await tool.run(path=path, content="hello world")
        assert not r.is_error
        assert "Written to" in r.output
        with open(path) as f:
            assert f.read() == "hello world"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_file_write_append():
    from synapsekit import FileWriteTool

    tool = FileWriteTool()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("line1\n")
        path = f.name

    try:
        r = await tool.run(path=path, content="line2\n", append=True)
        assert not r.is_error
        assert "Appended to" in r.output
        with open(path) as f:
            assert f.read() == "line1\nline2\n"
    finally:
        os.unlink(path)


@pytest.mark.asyncio
async def test_file_write_creates_dirs():
    from synapsekit import FileWriteTool

    tool = FileWriteTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sub", "dir", "file.txt")
        r = await tool.run(path=path, content="nested")
        assert not r.is_error
        with open(path) as f:
            assert f.read() == "nested"


# ------------------------------------------------------------------ #
# FileListTool
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_file_list():
    from synapsekit import FileListTool

    tool = FileListTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.txt"), "w").close()
        open(os.path.join(tmpdir, "b.py"), "w").close()
        os.makedirs(os.path.join(tmpdir, "subdir"))

        r = await tool.run(path=tmpdir)
        assert not r.is_error
        assert "a.txt" in r.output
        assert "b.py" in r.output
        assert "subdir/" in r.output


@pytest.mark.asyncio
async def test_file_list_pattern():
    from synapsekit import FileListTool

    tool = FileListTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "a.txt"), "w").close()
        open(os.path.join(tmpdir, "b.py"), "w").close()

        r = await tool.run(path=tmpdir, pattern="*.py")
        assert "b.py" in r.output
        assert "a.txt" not in r.output


@pytest.mark.asyncio
async def test_file_list_not_a_dir():
    from synapsekit import FileListTool

    tool = FileListTool()
    r = await tool.run(path="/nonexistent/path")
    assert r.is_error


# ------------------------------------------------------------------ #
# RegexTool
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_regex_findall():
    from synapsekit import RegexTool

    tool = RegexTool()
    r = await tool.run(pattern=r"\d+", text="abc 123 def 456")
    assert "123" in r.output
    assert "456" in r.output


@pytest.mark.asyncio
async def test_regex_replace():
    from synapsekit import RegexTool

    tool = RegexTool()
    r = await tool.run(pattern=r"\d+", text="abc 123 def 456", action="replace", replacement="NUM")
    assert r.output == "abc NUM def NUM"


@pytest.mark.asyncio
async def test_regex_search():
    from synapsekit import RegexTool

    tool = RegexTool()
    r = await tool.run(pattern=r"(\d+)-(\d+)", text="range: 10-20", action="search")
    assert "Found: 10-20" in r.output
    assert "Groups:" in r.output


@pytest.mark.asyncio
async def test_regex_split():
    from synapsekit import RegexTool

    tool = RegexTool()
    r = await tool.run(pattern=r",\s*", text="a, b, c", action="split")
    assert "a" in r.output
    assert "b" in r.output
    assert "c" in r.output


@pytest.mark.asyncio
async def test_regex_case_insensitive():
    from synapsekit import RegexTool

    tool = RegexTool()
    r = await tool.run(pattern="hello", text="Hello World", action="findall", flags="i")
    assert "Hello" in r.output


@pytest.mark.asyncio
async def test_regex_invalid_pattern():
    from synapsekit import RegexTool

    tool = RegexTool()
    r = await tool.run(pattern="[invalid", text="test")
    assert r.is_error
    assert "Invalid regex" in r.error


# ------------------------------------------------------------------ #
# JSONQueryTool
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_json_query_simple():
    from synapsekit import JSONQueryTool

    tool = JSONQueryTool()
    data = json.dumps({"name": "Alice", "age": 30})
    r = await tool.run(json_data=data, path="name")
    assert r.output == "Alice"


@pytest.mark.asyncio
async def test_json_query_nested():
    from synapsekit import JSONQueryTool

    tool = JSONQueryTool()
    data = json.dumps({"users": [{"name": "Alice"}, {"name": "Bob"}]})
    r = await tool.run(json_data=data, path="users.1.name")
    assert r.output == "Bob"


@pytest.mark.asyncio
async def test_json_query_returns_object():
    from synapsekit import JSONQueryTool

    tool = JSONQueryTool()
    data = json.dumps({"config": {"debug": True, "port": 8080}})
    r = await tool.run(json_data=data, path="config")
    parsed = json.loads(r.output)
    assert parsed["debug"] is True
    assert parsed["port"] == 8080


@pytest.mark.asyncio
async def test_json_query_invalid_path():
    from synapsekit import JSONQueryTool

    tool = JSONQueryTool()
    data = json.dumps({"name": "Alice"})
    r = await tool.run(json_data=data, path="nonexistent")
    assert r.is_error


@pytest.mark.asyncio
async def test_json_query_invalid_json():
    from synapsekit import JSONQueryTool

    tool = JSONQueryTool()
    r = await tool.run(json_data="not valid json", path="key")
    assert r.is_error
    assert "Invalid JSON" in r.error


# ------------------------------------------------------------------ #
# HTTPRequestTool — import only (no real HTTP in unit tests)
# ------------------------------------------------------------------ #


def test_http_request_tool_import():
    from synapsekit import HTTPRequestTool

    tool = HTTPRequestTool()
    assert tool.name == "http_request"


@pytest.mark.asyncio
async def test_http_request_no_url():
    from synapsekit import HTTPRequestTool

    tool = HTTPRequestTool()
    r = await tool.run()
    assert r.is_error
    assert "No URL" in r.error


# ------------------------------------------------------------------ #
# LLM providers — import and basic validation
# ------------------------------------------------------------------ #


def test_import_openrouter():
    from synapsekit import OpenRouterLLM

    assert OpenRouterLLM is not None


def test_import_together():
    from synapsekit import TogetherLLM

    assert TogetherLLM is not None


def test_import_fireworks():
    from synapsekit import FireworksLLM

    assert FireworksLLM is not None


def test_openrouter_base_url():
    pytest.importorskip("openai")
    from synapsekit import LLMConfig
    from synapsekit.llm.openrouter import _OPENROUTER_BASE_URL, OpenRouterLLM

    llm = OpenRouterLLM(LLMConfig(model="openai/gpt-4o", api_key="test"))
    assert llm._base_url == _OPENROUTER_BASE_URL


def test_together_base_url():
    pytest.importorskip("openai")
    from synapsekit import LLMConfig
    from synapsekit.llm.together import _TOGETHER_BASE_URL, TogetherLLM

    llm = TogetherLLM(LLMConfig(model="meta-llama/Llama-3-70b", api_key="test"))
    assert llm._base_url == _TOGETHER_BASE_URL


def test_fireworks_base_url():
    pytest.importorskip("openai")
    from synapsekit import LLMConfig
    from synapsekit.llm.fireworks import _FIREWORKS_BASE_URL, FireworksLLM

    llm = FireworksLLM(LLMConfig(model="accounts/fireworks/models/llama", api_key="test"))
    assert llm._base_url == _FIREWORKS_BASE_URL


def test_openrouter_custom_base_url():
    pytest.importorskip("openai")
    from synapsekit import LLMConfig
    from synapsekit.llm.openrouter import OpenRouterLLM

    llm = OpenRouterLLM(LLMConfig(model="test", api_key="test"), base_url="http://localhost:8000")
    assert llm._base_url == "http://localhost:8000"


# ------------------------------------------------------------------ #
# Facade auto-detection
# ------------------------------------------------------------------ #


def test_facade_openrouter_detection():
    from synapsekit.rag.facade import _make_llm

    # Model with slash should auto-detect as openrouter
    # We can't actually instantiate without openai, so just test the logic
    try:
        llm = _make_llm("openai/gpt-4o", "test-key", None, "sys", 0.2, 1024)
        # If openai is installed, check type
        assert type(llm).__name__ == "OpenRouterLLM"
    except ImportError:
        pytest.skip("openai not installed")


def test_facade_explicit_together():
    try:
        from synapsekit.rag.facade import _make_llm

        llm = _make_llm("meta-llama/Llama-3-70b", "test-key", "together", "sys", 0.2, 1024)
        assert type(llm).__name__ == "TogetherLLM"
    except ImportError:
        pytest.skip("openai not installed")


def test_facade_explicit_fireworks():
    try:
        from synapsekit.rag.facade import _make_llm

        llm = _make_llm("llama-v3-70b", "test-key", "fireworks", "sys", 0.2, 1024)
        assert type(llm).__name__ == "FireworksLLM"
    except ImportError:
        pytest.skip("openai not installed")


# ------------------------------------------------------------------ #
# Retrieval strategies — imports
# ------------------------------------------------------------------ #


def test_import_contextual_retriever():
    from synapsekit import ContextualRetriever

    assert ContextualRetriever is not None


def test_import_sentence_window_retriever():
    from synapsekit import SentenceWindowRetriever

    assert SentenceWindowRetriever is not None


# ------------------------------------------------------------------ #
# SentenceWindowRetriever — unit tests
# ------------------------------------------------------------------ #


def test_sentence_splitting():
    from synapsekit.retrieval.sentence_window import SentenceWindowRetriever

    sentences = SentenceWindowRetriever._split_sentences("Hello world. How are you? I am fine!")
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "How are you?"
    assert sentences[2] == "I am fine!"


def test_sentence_splitting_empty():
    from synapsekit.retrieval.sentence_window import SentenceWindowRetriever

    sentences = SentenceWindowRetriever._split_sentences("")
    assert sentences == []


# ------------------------------------------------------------------ #
# ContextualRetriever — prompt format test
# ------------------------------------------------------------------ #


def test_contextual_prompt_format():
    from synapsekit.retrieval.contextual import _CONTEXT_PROMPT

    formatted = _CONTEXT_PROMPT.format(chunk="test chunk")
    assert "test chunk" in formatted
    assert "<chunk>" in formatted


# ------------------------------------------------------------------ #
# Tool schemas
# ------------------------------------------------------------------ #


def test_tool_schemas():
    from synapsekit import (
        DateTimeTool,
        FileListTool,
        FileWriteTool,
        HTTPRequestTool,
        JSONQueryTool,
        RegexTool,
    )

    for tool_cls in [
        DateTimeTool,
        FileListTool,
        FileWriteTool,
        HTTPRequestTool,
        JSONQueryTool,
        RegexTool,
    ]:
        tool = tool_cls()
        schema = tool.schema()
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]

        anthropic = tool.anthropic_schema()
        assert "name" in anthropic
        assert "input_schema" in anthropic


# ------------------------------------------------------------------ #
# Version check
# ------------------------------------------------------------------ #


def test_version():
    import synapsekit

    assert synapsekit.__version__ == "1.0.0"
