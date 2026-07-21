"""AnthropicLLM HTTP-contract tests via respx (real SDK request-building + SSE parsing).

Replaces MagicMock coverage: respx intercepts the real httpx call the anthropic SDK
makes to the Messages API and streams back real event-typed SSE, exercising the
actual request body and stream/usage parsing. Part of #829 (LLM track).
"""

from __future__ import annotations

import inspect
import json

import httpx
import pytest

respx = pytest.importorskip("respx")
pytest.importorskip("anthropic")

from synapsekit.llm.anthropic import AnthropicLLM  # noqa: E402
from synapsekit.llm.base import LLMConfig  # noqa: E402

_URL = "https://api.anthropic.com/v1/messages"


def _event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _messages_sse(texts: list[str], input_tokens: int, output_tokens: int) -> bytes:
    parts = [
        _event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        ),
        _event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
    ]
    for t in texts:
        parts.append(
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": t},
                },
            )
        )
    parts += [
        _event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        _event("message_stop", {"type": "message_stop"}),
    ]
    return "".join(parts).encode()


def _llm() -> AnthropicLLM:
    return AnthropicLLM(
        LLMConfig(model="claude-3-5-sonnet", api_key="test-key", provider="anthropic")
    )


@pytest.mark.asyncio
@respx.mock
async def test_stream_parses_events_and_counts_tokens():
    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_messages_sse(["Hello", " world"], input_tokens=5, output_tokens=2),
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
            content=_messages_sse(["ok"], input_tokens=1, output_tokens=1),
        )
    )
    llm = _llm()
    _ = [t async for t in llm.stream("question", max_tokens=64)]

    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "claude-3-5-sonnet"
    assert sent["max_tokens"] == 64
    assert {"role": "user", "content": "question"} in sent["messages"]
    assert route.calls[0].request.headers["x-api-key"] == "test-key"


def test_stream_is_async_generator():
    assert inspect.isasyncgenfunction(AnthropicLLM.stream)
