"""Tests for built-in agent tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.agents.base import ToolResult
from synapsekit.agents.tools.calculator import CalculatorTool
from synapsekit.agents.tools.file_read import FileReadTool
from synapsekit.agents.tools.image_analysis import ImageAnalysisTool
from synapsekit.agents.tools.python_repl import PythonREPLTool
from synapsekit.agents.tools.speech_to_text import SpeechToTextTool
from synapsekit.agents.tools.sql_query import SQLQueryTool
from synapsekit.agents.tools.text_to_speech import TextToSpeechTool
from synapsekit.agents.tools.web_search import WebSearchTool

# ------------------------------------------------------------------ #
# ToolResult
# ------------------------------------------------------------------ #


class TestToolResult:
    def test_output_only(self):
        r = ToolResult(output="42")
        assert r.output == "42"
        assert r.error is None
        assert not r.is_error
        assert str(r) == "42"

    def test_error(self):
        r = ToolResult(output="", error="boom")
        assert r.is_error
        assert str(r) == "boom"


# ------------------------------------------------------------------ #
# BaseTool schema
# ------------------------------------------------------------------ #


class TestBaseToolSchema:
    def test_schema_format(self):
        calc = CalculatorTool()
        s = calc.schema()
        assert s["type"] == "function"
        assert s["function"]["name"] == "calculator"
        assert "parameters" in s["function"]

    def test_anthropic_schema_format(self):
        calc = CalculatorTool()
        s = calc.anthropic_schema()
        assert s["name"] == "calculator"
        assert "input_schema" in s


# ------------------------------------------------------------------ #
# CalculatorTool
# ------------------------------------------------------------------ #


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        r = await CalculatorTool().run(expression="2 + 2")
        assert r.output == "4"
        assert not r.is_error

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        r = await CalculatorTool().run(expression="2 ** 10")
        assert r.output == "1024"

    @pytest.mark.asyncio
    async def test_sqrt(self):
        r = await CalculatorTool().run(expression="sqrt(144)")
        assert float(r.output) == pytest.approx(12.0)

    @pytest.mark.asyncio
    async def test_pi(self):
        r = await CalculatorTool().run(expression="round(pi, 4)")
        assert r.output == "3.1416"

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        r = await CalculatorTool().run(expression="1 / 0")
        assert r.is_error
        assert "zero" in r.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        r = await CalculatorTool().run(expression="import os")
        assert r.is_error

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        r = await CalculatorTool().run(expression="")
        assert r.is_error

    @pytest.mark.asyncio
    async def test_input_kwarg_fallback(self):
        r = await CalculatorTool().run(input="3 * 3")
        assert r.output == "9"

    def test_name_and_description(self):
        t = CalculatorTool()
        assert t.name == "calculator"
        assert len(t.description) > 10


# ------------------------------------------------------------------ #
# PythonREPLTool
# ------------------------------------------------------------------ #


class TestPythonREPLTool:
    @pytest.mark.asyncio
    async def test_print_output(self):
        r = await PythonREPLTool().run(code="print('hello world')")
        assert "hello world" in r.output
        assert not r.is_error

    @pytest.mark.asyncio
    async def test_math_computation(self):
        r = await PythonREPLTool().run(code="print(sum(range(1, 6)))")
        assert "15" in r.output

    @pytest.mark.asyncio
    async def test_multiline_code(self):
        code = "x = 10\ny = 20\nprint(x + y)"
        r = await PythonREPLTool().run(code=code)
        assert "30" in r.output

    @pytest.mark.asyncio
    async def test_persistent_namespace(self):
        repl = PythonREPLTool()
        await repl.run(code="x = 42")
        r = await repl.run(code="print(x)")
        assert "42" in r.output

    @pytest.mark.asyncio
    async def test_reset_clears_namespace(self):
        repl = PythonREPLTool()
        await repl.run(code="x = 99")
        repl.reset()
        r = await repl.run(code="print(x)")
        assert r.is_error

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        r = await PythonREPLTool().run(code="def broken(:\n    pass")
        assert r.is_error

    @pytest.mark.asyncio
    async def test_no_output(self):
        r = await PythonREPLTool().run(code="x = 1 + 1")
        assert "no output" in r.output.lower()

    @pytest.mark.asyncio
    async def test_empty_code(self):
        r = await PythonREPLTool().run(code="")
        assert r.is_error


# ------------------------------------------------------------------ #
# FileReadTool
# ------------------------------------------------------------------ #


class TestFileReadTool:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello from file")
        r = await FileReadTool().run(path=str(f))
        assert r.output == "hello from file"
        assert not r.is_error

    @pytest.mark.asyncio
    async def test_missing_file(self):
        r = await FileReadTool().run(path="/nonexistent/file.txt")
        assert r.is_error
        assert "not found" in r.error.lower()

    @pytest.mark.asyncio
    async def test_empty_path(self):
        r = await FileReadTool().run(path="")
        assert r.is_error

    @pytest.mark.asyncio
    async def test_input_kwarg_fallback(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("content")
        r = await FileReadTool().run(input=str(f))
        assert r.output == "content"

    @pytest.mark.asyncio
    async def test_encoding_param(self, tmp_path):
        f = tmp_path / "utf8.txt"
        f.write_text("héllo", encoding="utf-8")
        r = await FileReadTool().run(path=str(f), encoding="utf-8")
        assert "héllo" in r.output


# ------------------------------------------------------------------ #
# WebSearchTool (mocked)
# ------------------------------------------------------------------ #


class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_import_error_without_duckduckgo(self):
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            tool = WebSearchTool()
            with pytest.raises(ImportError, match="duckduckgo-search"):
                await tool.run(query="test")

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_results = [
            {"title": "SynapseKit", "href": "https://example.com", "body": "Async RAG framework"},
            {"title": "Python", "href": "https://python.org", "body": "Programming language"},
        ]
        mock_ddgs = MagicMock()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = mock_results
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs.return_value = mock_ddgs_instance
        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs

        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            from synapsekit.agents.tools.web_search import WebSearchTool as WST

            tool = WST()
            r = await tool.run(query="SynapseKit")
            assert not r.is_error
            assert "SynapseKit" in r.output

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = WebSearchTool()
        r = await tool.run(query="")
        assert r.is_error

    def test_name_and_description(self):
        assert WebSearchTool.name == "web_search"


# ------------------------------------------------------------------ #
# SQLQueryTool
# ------------------------------------------------------------------ #


class TestSQLQueryTool:
    def _make_db(self, tmp_path):
        import sqlite3

        db = str(tmp_path / "test.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
        conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
        conn.commit()
        conn.close()
        return db

    @pytest.mark.asyncio
    async def test_select_query(self, tmp_path):
        db = self._make_db(tmp_path)
        tool = SQLQueryTool(db)
        r = await tool.run(query="SELECT * FROM users")
        assert not r.is_error
        assert "Alice" in r.output
        assert "Bob" in r.output

    @pytest.mark.asyncio
    async def test_filtered_select(self, tmp_path):
        db = self._make_db(tmp_path)
        tool = SQLQueryTool(db)
        r = await tool.run(query="SELECT name FROM users WHERE age > 28")
        assert "Alice" in r.output
        assert "Bob" not in r.output

    @pytest.mark.asyncio
    async def test_non_select_blocked(self, tmp_path):
        db = self._make_db(tmp_path)
        tool = SQLQueryTool(db)
        r = await tool.run(query="DROP TABLE users")
        assert r.is_error
        assert "SELECT" in r.error

    @pytest.mark.asyncio
    async def test_invalid_sql(self, tmp_path):
        db = self._make_db(tmp_path)
        tool = SQLQueryTool(db)
        r = await tool.run(query="SELECT * FROM nonexistent_table")
        assert r.is_error

    @pytest.mark.asyncio
    async def test_empty_query(self, tmp_path):
        db = self._make_db(tmp_path)
        tool = SQLQueryTool(db)
        r = await tool.run(query="")
        assert r.is_error

    @pytest.mark.asyncio
    async def test_markdown_table_format(self, tmp_path):
        db = self._make_db(tmp_path)
        tool = SQLQueryTool(db)
        r = await tool.run(query="SELECT id, name FROM users ORDER BY id")
        lines = r.output.strip().splitlines()
        assert "id" in lines[0]  # header
        assert "---" in lines[1]  # separator

    @pytest.mark.asyncio
    async def test_no_rows(self, tmp_path):
        db = self._make_db(tmp_path)
        tool = SQLQueryTool(db)
        r = await tool.run(query="SELECT * FROM users WHERE age > 100")
        assert "no rows" in r.output.lower()


# ------------------------------------------------------------------ #
# ImageAnalysisTool
# ------------------------------------------------------------------ #


class DummyLLM:
    def __init__(self, provider: str = "openai"):
        class _Config:
            def __init__(self, provider: str):
                self.provider = provider

        self.config = _Config(provider)

    async def generate_with_messages(self, messages, **kwargs):
        return "A cat sitting on a chair."


class TestImageAnalysisTool:
    @pytest.mark.asyncio
    async def test_requires_input(self):
        tool = ImageAnalysisTool(DummyLLM())
        r = await tool.run()
        assert r.is_error

    @pytest.mark.asyncio
    async def test_analyze_from_path(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"fake-image")
        tool = ImageAnalysisTool(DummyLLM())
        r = await tool.run(path=str(img))
        assert not r.is_error
        assert "cat" in r.output.lower()

    @pytest.mark.asyncio
    async def test_analyze_from_url(self):
        tool = ImageAnalysisTool(DummyLLM())
        r = await tool.run(image_url="https://example.com/image.png")
        assert not r.is_error


# ------------------------------------------------------------------ #
# TextToSpeechTool
# ------------------------------------------------------------------ #


class DummyTTSResponse:
    def stream_to_file(self, path):
        from pathlib import Path

        Path(path).write_bytes(b"audio-data")


class DummySpeech:
    def create(self, **kwargs):
        return DummyTTSResponse()


class DummyAudio:
    def __init__(self):
        self.speech = DummySpeech()


class DummyOpenAI:
    def __init__(self, api_key=None):
        self.audio = DummyAudio()


class TestTextToSpeechTool:
    @pytest.mark.asyncio
    async def test_requires_api_key(self, tmp_path):
        with patch.dict("os.environ", {}, clear=True):
            tool = TextToSpeechTool(api_key=None)
            r = await tool.run(text="hello", output_path=str(tmp_path / "out.mp3"))
            assert r.is_error

    @pytest.mark.asyncio
    async def test_synthesis(self, tmp_path):
        mock_module = MagicMock()
        mock_module.OpenAI = DummyOpenAI
        with patch.dict("sys.modules", {"openai": mock_module}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
                tool = TextToSpeechTool()
                out = tmp_path / "out.mp3"
                r = await tool.run(text="hello", output_path=str(out))
                assert not r.is_error
                assert out.exists()


# ------------------------------------------------------------------ #
# SpeechToTextTool
# ------------------------------------------------------------------ #


class DummyAudioLoader:
    def __init__(self, *args, **kwargs):
        pass

    async def aload(self):
        from synapsekit.loaders.base import Document

        return [Document(text="hello world")]


class TestSpeechToTextTool:
    @pytest.mark.asyncio
    async def test_requires_path(self):
        tool = SpeechToTextTool()
        r = await tool.run(path="")
        assert r.is_error

    @pytest.mark.asyncio
    async def test_transcription(self, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake-audio")
        with patch("synapsekit.agents.tools.speech_to_text.AudioLoader", DummyAudioLoader):
            tool = SpeechToTextTool()
            r = await tool.run(path=str(audio))
            assert not r.is_error
            assert "hello world" in r.output
