"""HTTP-contract tests for HuggingFaceEmbeddings (respx)."""

from __future__ import annotations

import httpx
import pytest

respx = pytest.importorskip("respx")

from synapsekit.embeddings.huggingface import HuggingFaceEmbeddings  # noqa: E402

_INFERENCE_URL = "https://api-inference.huggingface.co/models/BAAI/bge-base-en-v1.5"


def _list_payload() -> list[list[float]]:
    return [[0.1] * 4, [0.2] * 4]


@pytest.mark.asyncio
@respx.mock
async def test_inference_api_sends_inputs_and_parses_list():
    route = respx.post(_INFERENCE_URL).mock(return_value=httpx.Response(200, json=_list_payload()))
    emb = HuggingFaceEmbeddings(api_key="test-key")
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 4)
    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"
    assert route.calls[0].request.url.path.endswith("/models/BAAI/bge-base-en-v1.5")
    assert "inputs" in route.calls[0].request.content.decode()


@pytest.mark.asyncio
@respx.mock
async def test_tei_base_url_posts_to_embed():
    route = respx.post("http://localhost:8080/embed").mock(
        return_value=httpx.Response(200, json=_list_payload())
    )
    emb = HuggingFaceEmbeddings(api_key="test-key", base_url="http://localhost:8080")
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 4)
    assert route.calls[0].request.url.path == "/embed"


@pytest.mark.asyncio
@respx.mock
async def test_openai_style_response_supported():
    payload = {
        "data": [
            {"index": 0, "embedding": [0.1] * 4},
            {"index": 1, "embedding": [0.2] * 4},
        ]
    }
    respx.post(_INFERENCE_URL).mock(return_value=httpx.Response(200, json=payload))
    emb = HuggingFaceEmbeddings(api_key="test-key")
    vecs = await emb.embed(["a", "b"])
    assert vecs.shape == (2, 4)


def test_hf_token_takes_precedence_over_hf_api_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token-val")
    monkeypatch.setenv("HF_API_KEY", "key-val")
    emb = HuggingFaceEmbeddings()
    assert emb._get_key() == "token-val"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    emb = HuggingFaceEmbeddings(api_key=None)
    with pytest.raises(ValueError, match="HF_TOKEN"):
        emb._get_key()
