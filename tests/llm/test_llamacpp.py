from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.llamacpp import LlamaCppLLM


def make_config() -> LLMConfig:
    return LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp")


def _make_chunk(content: str | None) -> dict:
    delta = {"content": content} if content is not None else {}
    return {"choices": [{"delta": delta}]}


def _make_mock_model(chunks: list[dict]) -> MagicMock:
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = iter(chunks)
    return mock_model


class TestLlamaCppLLM:
    @pytest.fixture
    def llm(self) -> LlamaCppLLM:
        return LlamaCppLLM(make_config(), model_path="/models/test.gguf")

    # ------------------------------------------------------------------
    # Import error
    # ------------------------------------------------------------------

    def test_import_error_without_llama_cpp(self) -> None:
        llm = LlamaCppLLM(make_config(), model_path="/models/test.gguf")
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python"):
                llm._get_model()

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def test_model_loaded_lazily(self, llm: LlamaCppLLM) -> None:
        assert llm._model is None

    # ------------------------------------------------------------------
    # stream()
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, llm: LlamaCppLLM) -> None:
        llm._model = _make_mock_model(
            [
                _make_chunk("Hello"),
                _make_chunk(" world"),
            ]
        )

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self, llm: LlamaCppLLM) -> None:
        llm._model = _make_mock_model(
            [
                _make_chunk(""),
                _make_chunk(None),
                _make_chunk("Good"),
            ]
        )

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Good"]

    # ------------------------------------------------------------------
    # stream_with_messages()
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_with_messages(self, llm: LlamaCppLLM) -> None:
        llm._model = _make_mock_model([_make_chunk("Hi")])

        messages = [{"role": "user", "content": "hello"}]
        tokens = [t async for t in llm.stream_with_messages(messages)]
        assert tokens == ["Hi"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self, llm: LlamaCppLLM) -> None:
        llm._model = _make_mock_model([_make_chunk("x")])

        messages = [{"role": "user", "content": "q"}]
        _ = [t async for t in llm.stream_with_messages(messages, temperature=0.9)]

        call_kwargs = llm._model.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self, llm: LlamaCppLLM) -> None:
        llm._model = _make_mock_model([_make_chunk("x")])

        messages = [{"role": "user", "content": "q"}]
        _ = [t async for t in llm.stream_with_messages(messages, max_tokens=512)]

        call_kwargs = llm._model.create_chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_stream_passes_top_p(self, llm: LlamaCppLLM) -> None:
        llm._model = _make_mock_model([_make_chunk("x")])

        messages = [{"role": "user", "content": "q"}]
        _ = [t async for t in llm.stream_with_messages(messages, top_p=0.8)]

        call_kwargs = llm._model.create_chat_completion.call_args[1]
        assert call_kwargs["top_p"] == 0.8

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generate_collects_tokens(self, llm: LlamaCppLLM) -> None:
        llm._model = _make_mock_model(
            [
                _make_chunk("Hello"),
                _make_chunk(" world"),
            ]
        )

        result = await llm.generate("hi")
        assert result == "Hello world"

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_output_tokens_tracked(self, llm: LlamaCppLLM) -> None:
        llm._model = _make_mock_model(
            [
                _make_chunk("A"),
                _make_chunk("B"),
                _make_chunk("C"),
            ]
        )

        _ = [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 3
