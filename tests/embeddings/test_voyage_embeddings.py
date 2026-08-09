"""HTTP-contract tests for VoyageEmbeddings (respx)."""

from __future__ import annotations

import json

import httpx
import pytest

respx = pytest.importorskip("respx")

from synapsekit.embeddings.voyage import VoyageEmbeddings  # noqa: E402

_URL = "https://api.voyageai.com/v1/embeddings"


def _payload() -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1] * 4},
            {"object": "embedding", "index": 1, "embedding": [0.2] * 4},
        ],
        "model": "voyage-3",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


@pytest.mark.asyncio
@respx.mock
async def test_embed_parses_response():
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = VoyageEmbeddings(api_key="test-key")
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 4)
    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "voyage-3"
    assert body["input"] == ["a", "b"]


@pytest.mark.asyncio
@respx.mock
async def test_embed_one_returns_first_row():
    respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = VoyageEmbeddings(api_key="test-key")
    vec = await emb.embed_one("a")
    assert vec.shape == (4,)


@pytest.mark.asyncio
@respx.mock
async def test_non_200_raises_runtime_error():
    respx.post(_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    emb = VoyageEmbeddings(api_key="test-key")
    with pytest.raises(RuntimeError, match="401"):
        await emb.embed(["a"])
