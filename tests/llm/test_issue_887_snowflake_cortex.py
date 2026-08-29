from __future__ import annotations

import inspect
import json

import httpx
import pytest

respx = pytest.importorskip("respx")

from synapsekit.llm.base import LLMConfig  # noqa: E402
from synapsekit.llm.snowflake_cortex import SnowflakeCortexLLM  # noqa: E402

CUSTOM_URL = "https://snowflake.example.test/api/v2/cortex/v1/chat/completions"
ACCOUNT_URL = "https://acme-account.snowflakecomputing.com/api/v2/cortex/v1/chat/completions"


def _config(*, api_key: str = "snowflake-token") -> LLMConfig:
    return LLMConfig(
        model="mistral-large2",
        api_key=api_key,
        provider="snowflake-cortex",
        system_prompt="Answer directly.",
        max_retries=0,
    )


def _event(
    *, content: str | None = None, messages: str | None = None, usage: dict | None = None
) -> dict:
    choice: dict = {"delta": {}}
    if content is not None:
        choice["delta"]["content"] = content
    if messages is not None:
        choice = {"messages": messages}
    chunk: dict = {"choices": [choice]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _sse(*events: dict) -> bytes:
    return (
        "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    ).encode()


@pytest.mark.asyncio
@respx.mock
async def test_cortex_posts_to_custom_endpoint_with_bearer_and_token_type():
    route = respx.post(CUSTOM_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                _event(content="Hello"),
                _event(content=" Cortex", usage={"prompt_tokens": 5, "completion_tokens": 2}),
            ),
        )
    )
    llm = SnowflakeCortexLLM(
        _config(),
        base_url="https://snowflake.example.test/",
        authorization_token_type="SESSION_TOKEN",
    )

    result = "".join(
        [token async for token in llm.stream("question", temperature=0.8, max_tokens=21, top_p=0.9)]
    )

    assert result == "Hello Cortex"
    assert route.called
    sent = json.loads(route.calls[0].request.content)
    assert sent == {
        "model": "mistral-large2",
        "messages": [
            {"role": "system", "content": "Answer directly."},
            {"role": "user", "content": "question"},
        ],
        "stream": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_completion_tokens": 21,
    }
    assert route.calls[0].request.headers["authorization"] == "Bearer snowflake-token"
    assert route.calls[0].request.headers["x-snowflake-authorization-token-type"] == "SESSION_TOKEN"
    assert llm.tokens_used == {"input": 5, "output": 2}


@pytest.mark.asyncio
@respx.mock
async def test_cortex_uses_account_identifier_to_build_default_endpoint_and_default_header():
    route = respx.post(ACCOUNT_URL).mock(
        return_value=httpx.Response(200, json={"choices": [{"messages": "ok"}]})
    )
    llm = SnowflakeCortexLLM(_config(), account_identifier="acme-account")

    assert await llm.generate("hi") == "ok"
    assert route.called
    assert (
        route.calls[0].request.headers["x-snowflake-authorization-token-type"]
        == "PROGRAMMATIC_ACCESS_TOKEN"
    )


@pytest.mark.asyncio
@respx.mock
async def test_cortex_parses_documented_json_messages_and_usage():
    route = respx.post(CUSTOM_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"messages": [{"role": "assistant", "content": "json answer"}]}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            },
        )
    )
    llm = SnowflakeCortexLLM(_config(), base_url="https://snowflake.example.test")

    assert await llm.generate("hi") == "json answer"
    assert llm.tokens_used == {"input": 8, "output": 4}
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_cortex_raises_http_errors_and_rejects_missing_endpoint_or_key_before_network():
    llm = SnowflakeCortexLLM(_config(api_key=""))
    with pytest.raises(ValueError, match="api_key"):
        _ = [token async for token in llm.stream("hi")]
    assert not respx.calls

    route = respx.post(CUSTOM_URL).mock(return_value=httpx.Response(429, text="rate limited"))
    llm = SnowflakeCortexLLM(_config(), base_url="https://snowflake.example.test")
    with pytest.raises(httpx.HTTPStatusError):
        _ = [token async for token in llm.stream("hi")]
    assert route.called

    llm = SnowflakeCortexLLM(_config())
    with pytest.raises(ValueError, match="account_identifier or base_url"):
        _ = [token async for token in llm.stream("hi")]


def test_cortex_stream_is_async_generator_function():
    assert inspect.isasyncgenfunction(SnowflakeCortexLLM.stream)


@pytest.mark.asyncio
@respx.mock
async def test_cortex_accumulates_usage_and_closes_client():
    route = respx.post(CUSTOM_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                _event(content="Hello", usage={"prompt_tokens": 5, "completion_tokens": 2}),
                _event(content=" world", usage={"prompt_tokens": 7, "completion_tokens": 3}),
            ),
        )
    )
    llm = SnowflakeCortexLLM(_config(), base_url="https://snowflake.example.test")
    await llm.stream("hi").__anext__()
    await llm.stream("hi again").__anext__()
    assert route.call_count == 2
    assert llm.tokens_used == {"input": 10, "output": 4}
    await llm.aclose()
    assert llm._client is None


@pytest.mark.asyncio
@respx.mock
async def test_cortex_supports_async_context_manager():
    respx.post(CUSTOM_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(_event(content="ok")),
        )
    )
    async with SnowflakeCortexLLM(_config(), base_url="https://snowflake.example.test") as llm:
        assert "".join([token async for token in llm.stream("hi")]) == "ok"
    assert llm._client is None
