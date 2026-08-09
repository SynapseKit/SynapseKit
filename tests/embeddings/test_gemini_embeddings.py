"""HTTP-contract tests for GeminiEmbeddings (mocked client)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


class _FakeGenAI(ModuleType):
    """Fake ``google.genai`` module exposing ``Client``."""

    Client = MagicMock()


class _Embedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


@pytest.fixture
def fake_genai():
    """Install a fake ``google.genai`` module for the duration of a test."""
    fake = _FakeGenAI("google.genai")
    fake.Client = MagicMock()
    with patch.dict(
        "sys.modules",
        {"google.genai": fake, "google": sys.modules.get("google", ModuleType("google"))},
    ):
        yield fake


def _make_client(resp) -> MagicMock:
    client = MagicMock()
    client.models.embed_content = AsyncMock(return_value=resp)
    return client


@pytest.mark.asyncio
async def test_embed_parses_values(fake_genai):
    resp = MagicMock()
    resp.embeddings = [_Embedding([0.1] * 4), _Embedding([0.2] * 4)]
    mock_client = _make_client(resp)
    fake_genai.Client.return_value = mock_client

    from synapsekit.embeddings.gemini import GeminiEmbeddings

    emb = GeminiEmbeddings(api_key="test-key")
    vecs = await emb.embed(["a", "b"])

    assert vecs.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0)
    kwargs = mock_client.models.embed_content.call_args.kwargs
    assert kwargs["model"] == "text-embedding-004"
    assert kwargs["contents"] == ["a", "b"]


@pytest.mark.asyncio
async def test_embed_one_returns_first_row(fake_genai):
    resp = MagicMock()
    resp.embeddings = [_Embedding([0.3] * 4)]
    mock_client = _make_client(resp)
    fake_genai.Client.return_value = mock_client

    from synapsekit.embeddings.gemini import GeminiEmbeddings

    emb = GeminiEmbeddings(api_key="test-key")
    vec = await emb.embed_one("a")

    assert vec.shape == (4,)


def test_missing_key_raises(fake_genai):
    from synapsekit.embeddings.gemini import GeminiEmbeddings

    emb = GeminiEmbeddings(api_key=None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        with patch.dict("synapsekit.embeddings.gemini.os.environ", {}, clear=True):
            emb._get_client()
