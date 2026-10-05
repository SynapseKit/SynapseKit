"""HTTP-contract tests for MistralLLM (respx)."""

from __future__ import annotations

import json

import httpx
import pytest

respx = pytest.importorskip("respx")
pytest.importorskip("mistralai")

from synapsekit.llm.base import LLMConfig  # noqa: E402
from synapsekit.llm.mistral import MistralLLM  # noqa: E402

_URL = "https://api.mistral.ai/v1/chat/completions"


def _sse(*texts: str) -> bytes:
    chunks = [
        {
            "id": "cmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mistral-small-latest",
            "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}],
        }
        for t in texts
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


@pytest.mark.asyncio
@respx.mock
async def test_stream_parses_sse():
    route = respx.post(_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse("Hello", " world")
        )
    )
    llm = MistralLLM(
        LLMConfig(model="mistral-small-latest", api_key="test-key", provider="mistral")
    )
    out = [t async for t in llm.stream("hi", max_tokens=50)]

    assert out == ["Hello", " world"]
    assert llm._output_tokens == 2
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "mistral-small-latest"
    assert sent["stream"] is True
    assert sent["max_tokens"] == 50
    assert sent["messages"][-1] == {"role": "user", "content": "hi"}
    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"
