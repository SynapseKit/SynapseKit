"""HTTP-contract tests for OpenAIEmbeddings (respx)."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

respx = pytest.importorskip("respx")
pytest.importorskip("openai")

from tests.embeddings.conftest import embed_payload  # noqa: E402

from synapsekit.embeddings.openai import OpenAIEmbeddings  # noqa: E402

_URL = "https://api.openai.com/v1/embeddings"


def _emb() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(api_key="test-key")


@pytest.mark.asyncio
@respx.mock
async def test_embed_parses_response_and_sends_bearer():
    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200, json=embed_payload("text-embedding-3-small", 8, ["a", "b"])
        )
    )
    emb = _emb()
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 8)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == ["a", "b"]


@pytest.mark.asyncio
@respx.mock
async def test_embed_dimensions_param_forwarded_when_set():
    route = respx.post(_URL).mock(
        return_value=httpx.Response(200, json=embed_payload("text-embedding-3-small", 8, ["a"]))
    )
    emb = OpenAIEmbeddings(api_key="test-key", dimensions=8)
    await emb.embed(["a"])
    body = json.loads(route.calls[0].request.content)
    assert body["dimensions"] == 8


@pytest.mark.asyncio
@respx.mock
async def test_embed_one_matches_first_row():
    respx.post(_URL).mock(
        return_value=httpx.Response(200, json=embed_payload("text-embedding-3-small", 8, ["a"]))
    )
    emb = _emb()
    vec = await emb.embed_one("a")
    assert vec.shape == (8,)


@pytest.mark.asyncio
@respx.mock
async def test_embed_reorders_by_index():
    payload = embed_payload("text-embedding-3-small", 4, ["a", "b"])
    payload["data"] = [payload["data"][1], payload["data"][0]]
    respx.post(_URL).mock(return_value=httpx.Response(200, json=payload))
    emb = _emb()
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 4)
