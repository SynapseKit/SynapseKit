"""HTTP-contract tests for JinaEmbeddings (respx)."""

from __future__ import annotations

import json

import httpx
import pytest

respx = pytest.importorskip("respx")

from synapsekit.embeddings.jina import JinaEmbeddings  # noqa: E402

_URL = "https://api.jina.ai/v1/embeddings"


def _payload() -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1] * 4},
            {"object": "embedding", "index": 1, "embedding": [0.2] * 4},
        ],
        "model": "jina-embeddings-v3",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


@pytest.mark.asyncio
@respx.mock
async def test_embed_sends_task_and_dimensions():
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = JinaEmbeddings(api_key="test-key", task="retrieval.passage", dimensions=4)
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 4)
    body = json.loads(route.calls[0].request.content)
    assert body["task"] == "retrieval.passage"
    assert body["dimensions"] == 4
    assert body["model"] == "jina-embeddings-v3"
    assert body["input"] == ["a", "b"]


@pytest.mark.asyncio
@respx.mock
async def test_embed_one_returns_first_row():
    respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = JinaEmbeddings(api_key="test-key")
    vec = await emb.embed_one("a")
    assert vec.shape == (4,)
