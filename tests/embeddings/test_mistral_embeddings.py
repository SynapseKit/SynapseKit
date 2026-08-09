"""HTTP-contract tests for MistralEmbeddings (respx)."""

from __future__ import annotations

import json

import httpx
import pytest

respx = pytest.importorskip("respx")
pytest.importorskip("mistralai")

from synapsekit.embeddings.mistral import MistralEmbeddings  # noqa: E402

_URL = "https://api.mistral.ai/v1/embeddings"


def _payload() -> dict:
    return {
        "id": "emb-1",
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1] * 4},
            {"object": "embedding", "index": 1, "embedding": [0.2] * 4},
        ],
        "model": "mistral-embed",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


@pytest.mark.asyncio
@respx.mock
async def test_embed_parses_response():
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = MistralEmbeddings(api_key="test-key")
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 4)
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "mistral-embed"
    assert body["inputs"] == ["a", "b"]


@pytest.mark.asyncio
@respx.mock
async def test_embed_one_returns_first_row():
    respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = MistralEmbeddings(api_key="test-key")
    vec = await emb.embed_one("a")
    assert vec.shape == (4,)
