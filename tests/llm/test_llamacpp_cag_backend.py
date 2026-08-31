from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.llamacpp import LlamaCppLLM
from synapsekit.llm._llamacpp_cag_backend import LlamaCppCAGBackend


def make_config() -> LLMConfig:
    return LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp")


class TestLlamaCppCAGBackend:
    @pytest.fixture
    def llm(self) -> LlamaCppLLM:
        return LlamaCppLLM(make_config(), model_path="/models/test.gguf")

    @pytest.fixture
    def backend(self) -> LlamaCppCAGBackend:
        return LlamaCppCAGBackend()

    def test_supports(self, backend: LlamaCppCAGBackend, llm: LlamaCppLLM) -> None:
        assert backend.supports(llm) is True
        
        # Test unsupported LLM provider
        unsupported = MagicMock()
        assert backend.supports(unsupported) is False

    def test_import_error_without_llama_cpp(self, backend: LlamaCppCAGBackend, llm: LlamaCppLLM) -> None:
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(ImportError, match="llama-cpp-python"):
                # Importing / retrieving model should trigger ImportError
                llm._get_model()

    @pytest.mark.asyncio
    async def test_build_cache(self, backend: LlamaCppCAGBackend, llm: LlamaCppLLM) -> None:
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3]
        mock_model.save_state.return_value = b"serialized state bytes"
        
        llm._model = mock_model

        res = await backend.build_cache(llm, "corpus text here")
        
        assert res["corpus_text"] == "corpus text here"
        assert res["state"] == b"serialized state bytes"
        mock_model.reset.assert_called_once()
        mock_model.tokenize.assert_called_once_with(b"corpus text here", add_bos=True)
        mock_model.eval.assert_called_once_with([1, 2, 3])
        mock_model.save_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_cache(self, backend: LlamaCppCAGBackend, llm: LlamaCppLLM) -> None:
        mock_model = MagicMock()
        # Mock streaming completion
        mock_model.create_completion.return_value = [
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
        ]
        
        llm._model = mock_model

        cache_handle = {
            "state": b"serialized state bytes",
            "corpus_text": "corpus text here",
        }

        tokens = []
        async for t in backend.generate_with_cache(llm, cache_handle, " query text"):
            tokens.append(t)

        assert tokens == ["Hello", " world"]
        mock_model.load_state.assert_called_once_with(b"serialized state bytes")
        mock_model.create_completion.assert_called_once_with(
            prompt="corpus text here query text",
            stream=True,
            temperature=llm.config.temperature,
            max_tokens=llm.config.max_tokens,
            top_p=llm._top_p,
        )
