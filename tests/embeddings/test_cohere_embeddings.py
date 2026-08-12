"""HTTP-contract tests for CohereEmbeddings (respx)."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

respx = pytest.importorskip("respx")
pytest.importorskip("cohere")

from synapsekit.embeddings.cohere import CohereEmbeddings  # noqa: E402

_URL = "https://api.cohere.com/v2/embed"


def _payload() -> dict:
    return {
        "embeddings": {"float": [[0.1] * 4, [0.2] * 4]},
        "texts": ["a", "b"],
    }


@pytest.mark.asyncio
@respx.mock
async def test_embed_sends_document_input_type():
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = CohereEmbeddings(api_key="test-key")
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 4)
    body = json.loads(route.calls[0].request.content)
    assert body["input_type"] == "search_document"
    assert body["model"] == "embed-v3"
    assert body["texts"] == ["a", "b"]


@pytest.mark.asyncio
@respx.mock
async def test_embed_one_sends_query_input_type():
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = CohereEmbeddings(api_key="test-key")
    vec = await emb.embed_one("a")
    assert vec.shape == (4,)
    body = json.loads(route.calls[0].request.content)
    assert body["input_type"] == "search_query"


@pytest.mark.asyncio
@respx.mock
async def test_custom_input_types_respected():
    route = respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = CohereEmbeddings(
        api_key="test-key", input_type="classification", query_input_type="clustering"
    )
    await emb.embed(["a"])
    await emb.embed_one("a")
    assert json.loads(route.calls[0].request.content)["input_type"] == "classification"
    assert json.loads(route.calls[1].request.content)["input_type"] == "clustering"


@pytest.mark.asyncio
@respx.mock
async def test_embed_normalizes():
    respx.post(_URL).mock(return_value=httpx.Response(200, json=_payload()))
    emb = CohereEmbeddings(api_key="test-key")
    vecs = await emb.embed(["a", "b"])
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0)
