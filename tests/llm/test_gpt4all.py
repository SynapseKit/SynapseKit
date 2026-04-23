from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.gpt4all import GPT4AllLLM


def make_config() -> LLMConfig:
    return LLMConfig(model="orca-mini-3b-gguf2-q4_0.gguf", api_key="", provider="gpt4all")


def _configure_mock_generate(mock_model: MagicMock, chunks: list[str]) -> None:
    def _generate(_prompt: str, **kwargs):
        callback = kwargs.get("callback")
        if callback is not None:
            for i, text in enumerate(chunks):
                callback(i, text)
        return "".join(chunks)

    mock_model.generate.side_effect = _generate


def _configure_mock_generate_iter(mock_model: MagicMock, chunks: list[str]) -> None:
    def _generate(_prompt: str, **kwargs):
        callback = kwargs.get("callback")

        def _iter():
            for i, text in enumerate(chunks):
                if callback is not None:
                    callback(i, text)
                yield text

        return _iter()

    mock_model.generate.side_effect = _generate


class TestGPT4AllLLM:
    def test_import_error_without_gpt4all(self) -> None:
        llm = GPT4AllLLM(make_config())
        with patch.dict("sys.modules", {"gpt4all": None}):
            with pytest.raises(ImportError, match="gpt4all"):
                llm._get_model()

    def test_model_loaded_lazily(self) -> None:
        llm = GPT4AllLLM(make_config())
        assert llm._model is None

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self) -> None:
        llm = GPT4AllLLM(make_config())
        llm._model = MagicMock()
        _configure_mock_generate(llm._model, ["Hello", " world"])

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_content(self) -> None:
        llm = GPT4AllLLM(make_config())
        llm._model = MagicMock()
        _configure_mock_generate(llm._model, ["", "Good"])

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Good"]

    @pytest.mark.asyncio
    async def test_stream_with_messages(self) -> None:
        llm = GPT4AllLLM(make_config())
        llm._model = MagicMock()
        _configure_mock_generate(llm._model, ["Hi"])

        messages = [{"role": "user", "content": "hello"}]
        tokens = [t async for t in llm.stream_with_messages(messages)]
        assert tokens == ["Hi"]

    @pytest.mark.asyncio
    async def test_stream_with_iterable_generate_and_callback_no_duplicates(self) -> None:
        llm = GPT4AllLLM(make_config())
        llm._model = MagicMock()
        _configure_mock_generate_iter(llm._model, ["Hello", " world"])

        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_passes_temperature(self) -> None:
        llm = GPT4AllLLM(make_config())
        llm._model = MagicMock()
        _configure_mock_generate(llm._model, ["x"])

        _ = [
            t
            async for t in llm.stream_with_messages(
                [{"role": "user", "content": "q"}], temperature=0.9
            )
        ]

        call_kwargs = llm._model.generate.call_args[1]
        assert call_kwargs["temp"] == 0.9

    @pytest.mark.asyncio
    async def test_stream_passes_max_tokens(self) -> None:
        llm = GPT4AllLLM(make_config())
        llm._model = MagicMock()
        _configure_mock_generate(llm._model, ["x"])

        _ = [
            t
            async for t in llm.stream_with_messages(
                [{"role": "user", "content": "q"}],
                max_tokens=512,
            )
        ]

        call_kwargs = llm._model.generate.call_args[1]
        assert call_kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_generate_collects_tokens(self) -> None:
        llm = GPT4AllLLM(make_config())
        llm._model = MagicMock()
        _configure_mock_generate(llm._model, ["Hello", " world"])

        result = await llm.generate("hi")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_output_tokens_tracked(self) -> None:
        llm = GPT4AllLLM(make_config())
        llm._model = MagicMock()
        _configure_mock_generate(llm._model, ["A", "B", "C"])

        _ = [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 3

    def test_import_from_top_level(self) -> None:
        from synapsekit import GPT4AllLLM as TopLevelGPT4AllLLM

        assert TopLevelGPT4AllLLM is not None


def test_facade_explicit_gpt4all() -> None:
    from synapsekit.llm._factory import make_llm

    mock_gpt4all = MagicMock()
    with patch.dict("sys.modules", {"gpt4all": mock_gpt4all}):
        llm = make_llm("orca-mini-3b-gguf2-q4_0.gguf", "", "gpt4all", "sys", 0.2, 100)
        assert isinstance(llm, GPT4AllLLM)

