from __future__ import annotations

import inspect
import json

import httpx
import pytest

respx = pytest.importorskip("respx")

from synapsekit.llm.base import LLMConfig  # noqa: E402
from synapsekit.llm.nvidia_nim import NvidiaNIMLLM  # noqa: E402

DEFAULT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
CUSTOM_URL = "https://nim.example.test/v1/chat/completions"


def _config(*, api_key: str = "nim-secret") -> LLMConfig:
    return LLMConfig(
        model="meta/llama-3.1-8b-instruct",
        api_key=api_key,
        provider="nvidia-nim",
        system_prompt="Be concise.",
        max_retries=0,
    )


def _chunk(*, content: str | None = None, usage: dict | None = None) -> dict:
    chunk: dict = {
        "id": "nim-1",
        "object": "chat.completion.chunk",
        "choices": [] if content is None else [{"delta": {"content": content}}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _sse(*chunks: dict) -> bytes:
    return (
        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    ).encode()


def _response(*chunks: dict) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_sse(*chunks),
    )


@pytest.mark.asyncio
@respx.mock
async def test_nim_posts_to_custom_endpoint_with_bearer_and_request_parameters():
    route = respx.post(CUSTOM_URL).mock(
        return_value=_response(_chunk(content="Hello"), _chunk(content=" world"))
    )
    llm = NvidiaNIMLLM(_config(), base_url="https://nim.example.test/v1")

    result = "".join(
        [token async for token in llm.stream("question", temperature=0.7, max_tokens=33, top_p=0.9)]
    )

    assert result == "Hello world"
    assert route.called
    sent = json.loads(route.calls[0].request.content)
    assert sent == {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "question"},
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 33,
        "top_p": 0.9,
    }
    assert route.calls[0].request.headers["authorization"] == "Bearer nim-secret"


@pytest.mark.asyncio
@respx.mock
async def test_nim_uses_hosted_default_and_omits_authorization_without_key():
    route = respx.post(DEFAULT_URL).mock(return_value=_response(_chunk(content="ok")))
    llm = NvidiaNIMLLM(_config(api_key=""))

    assert "".join([token async for token in llm.stream("hi")]) == "ok"

    assert route.called
    assert "authorization" not in route.calls[0].request.headers


@pytest.mark.asyncio
@respx.mock
async def test_nim_parses_usage_and_ignores_malformed_or_after_done_events():
    route = respx.post(DEFAULT_URL).mock(
        return_value=_response(
            {"not": "json"},
            _chunk(content="a"),
            _chunk(content="b", usage={"prompt_tokens": 4, "completion_tokens": 2}),
        )
    )
    llm = NvidiaNIMLLM(_config())

    tokens = [token async for token in llm.stream("hi")]

    assert tokens == ["a", "b"]
    assert llm.tokens_used == {"input": 4, "output": 2}
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_nim_raises_http_errors_from_streaming_response():
    route = respx.post(DEFAULT_URL).mock(return_value=httpx.Response(503, text="unavailable"))
    llm = NvidiaNIMLLM(_config())

    with pytest.raises(httpx.HTTPStatusError):
        _ = [token async for token in llm.stream("hi")]

    assert route.called


def test_nim_stream_is_async_generator_function_and_endpoint_is_normalized():
    assert inspect.isasyncgenfunction(NvidiaNIMLLM.stream)
    assert (
        NvidiaNIMLLM(_config(), base_url="https://nim.example.test/v1/")._endpoint() == CUSTOM_URL
    )
