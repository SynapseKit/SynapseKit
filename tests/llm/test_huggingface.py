from unittest.mock import AsyncMock, patch

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.huggingface import HuggingFaceLLM

try:
    import huggingface_hub  # noqa: F401

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


@pytest.fixture
def mock_hf_client():
    if not HUGGINGFACE_AVAILABLE:
        yield None
        return

    with patch("synapsekit.llm.huggingface.AsyncInferenceClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.mark.skipif(not HUGGINGFACE_AVAILABLE, reason="huggingface-hub not installed")
def test_huggingface_init():
    config = LLMConfig(provider="huggingface", model="meta-llama/Llama-2-7b", api_key="test_key")
    llm = HuggingFaceLLM(config)
    assert llm.client.model == "meta-llama/Llama-2-7b"
    assert llm.client.token == "test_key"


@pytest.mark.skipif(not HUGGINGFACE_AVAILABLE, reason="huggingface-hub not installed")
@pytest.mark.asyncio
async def test_huggingface_generate(mock_hf_client):
    mock_hf_client.text_generation.return_value = "Hello from Hugging Face!"

    config = LLMConfig(
        provider="huggingface", model="test-model", api_key="test_key", max_tokens=100
    )
    llm = HuggingFaceLLM(config)

    response = await llm._generate("Say hello", system="You are an AI.")

    assert response == "Hello from Hugging Face!"
    mock_hf_client.text_generation.assert_called_once_with(
        prompt="You are an AI.\n\nSay hello",
        max_new_tokens=100,
        temperature=0.2,
        top_p=0.95,
        stream=False,
    )


@pytest.mark.skipif(not HUGGINGFACE_AVAILABLE, reason="huggingface-hub not installed")
@pytest.mark.asyncio
async def test_huggingface_stream(mock_hf_client):
    # Mocking the async generator returned by text_generation(stream=True)
    async def mock_stream_response():
        for chunk in ["Hello", " from", " Hugging", " Face!"]:
            yield chunk

    mock_hf_client.text_generation.return_value = mock_stream_response()

    config = LLMConfig(provider="huggingface", model="test-model", api_key="test_key")
    llm = HuggingFaceLLM(config)

    chunks = []
    async for chunk in llm.stream("test prompt"):
        chunks.append(chunk)

    assert chunks == ["Hello", " from", " Hugging", " Face!"]
    mock_hf_client.text_generation.assert_called_once_with(
        prompt="test prompt", max_new_tokens=1024, temperature=0.2, top_p=0.95, stream=True
    )
