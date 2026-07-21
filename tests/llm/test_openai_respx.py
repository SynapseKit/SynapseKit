"""OpenAILLM HTTP-contract tests via respx (real SDK request-building + SSE parsing).

Replaces MagicMock coverage: respx intercepts the real httpx call the openai SDK
makes and streams back real SSE, so this exercises the actual request body the SDK
builds and the actual chunk parsing / token accounting — not a hand-faked client.
Part of #829 (LLM track).
"""

from __future__ import annotations

import inspect
import json

import httpx
import pytest

respx = pytest.importorskip("respx")
pytest.importorskip("openai")

from synapsekit.llm.base import LLMConfig  # noqa: E402
from synapsekit.llm.openai import OpenAILLM  # noqa: E402

_URL = "https://api.openai.com/v1/chat/completions"


def _sse(*chunks: dict) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def _chunk(delta: dict | None = None, usage: dict | None = None) -> dict:
    c: dict = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    if usage is not None:
        c["usage"] = usage
    return c


def _llm() -> OpenAILLM:
    return OpenAILLM(LLMConfig(model="gpt-4o-mini", api_key="test-key", provider="openai"))


@pytest.mark.asyncio
@respx.mock
async def test_stream_parses_sse_and_counts_tokens():
    body = _sse(
        _chunk({"content": "Hello"}),
        _chunk({"content": " world"}),
        _chunk(usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
    )
    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )
    )
    llm = _llm()
    out = "".join([t async for t in llm.stream("hi there")])

    assert out == "Hello world"
    assert route.called
    assert llm._input_tokens == 5
    assert llm._output_tokens == 2


@pytest.mark.asyncio
@respx.mock
async def test_request_body_is_built_correctly():
    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(_chunk({"content": "ok"})),
        )
    )
    llm = _llm()
    _ = [t async for t in llm.stream("question", temperature=0.7, max_tokens=50)]

    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "gpt-4o-mini"
    assert sent["stream"] is True
    assert sent["temperature"] == 0.7
    assert sent["max_tokens"] == 50
    assert {"role": "user", "content": "question"} in sent["messages"]
    # API key is sent as a bearer token, proving real auth-header construction.
    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"


def test_stream_is_async_generator():
    # SynapseKit is async-first.
    assert inspect.isasyncgenfunction(OpenAILLM.stream)
