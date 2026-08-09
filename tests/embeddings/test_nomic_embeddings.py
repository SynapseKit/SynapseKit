"""HTTP-contract tests for NomicEmbeddings (respx)."""

from __future__ import annotations

import json

import httpx
import pytest

respx = pytest.importorskip("respx")

from synapsekit.embeddings.nomic import NomicEmbeddings  # noqa: E402

_URL = "https://api.nomic.ai/v1/embeddings"


def _payload() -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1] * 4},
            {"object": "embedding", "index": 1, "embedding": [0.2] * 4},
        ],
        "model": "nomic-embed-text",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }


@pytest.mark.asyncio
@respx.mock
async def test_embed_sends_document_task_type():
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = NomicEmbeddings(api_key="test-key")
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 4)
    body = json.loads(route.calls[0].request.content)
    assert body["task_type"] == "search_document"
    assert body["model"] == "nomic-embed-text"


@pytest.mark.asyncio
@respx.mock
async def test_embed_one_sends_query_task_type():
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = NomicEmbeddings(api_key="test-key")
    vec = await emb.embed_one("a")
    assert vec.shape == (4,)
    body = json.loads(route.calls[0].request.content)
    assert body["task_type"] == "search_query"


@pytest.mark.asyncio
@respx.mock
async def test_embed_one_restores_task_type_after_call():
    respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = NomicEmbeddings(api_key="test-key")
    await emb.embed_one("a")
    assert emb._request_extra["task_type"] == "search_document"
