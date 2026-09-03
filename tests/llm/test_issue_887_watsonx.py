from __future__ import annotations

import inspect
import json
import time
from urllib.parse import parse_qs

import httpx
import pytest

respx = pytest.importorskip("respx")

from synapsekit.llm.base import LLMConfig  # noqa: E402
from synapsekit.llm.watsonx import WatsonxLLM  # noqa: E402

IAM_URL = "https://iam.cloud.ibm.com/identity/token"
# watsonx.ai streams from the dedicated ``chat_stream`` endpoint (plain ``chat``
# is non-streaming and ignores a ``stream`` body field).
DEFAULT_API_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat_stream?version=2024-10-01"
CUSTOM_API_URL = "https://watson.example.test/ml/v1/text/chat_stream?version=2025-01-15"


def _config(*, api_key: str = "ibm-api-key") -> LLMConfig:
    return LLMConfig(
        model="ibm/granite-3-8b-instruct",
        api_key=api_key,
        provider="watsonx",
        system_prompt="You are terse.",
        max_retries=0,
    )


def _token_response(token: str = "iam-token", *, expires_in: int = 3600) -> httpx.Response:
    return httpx.Response(
        200,
        json={"access_token": token, "token_type": "Bearer", "expires_in": expires_in},
    )


def _api_response(*, results: list[dict], usage: dict | None = None) -> httpx.Response:
    payload: dict = {"results": results}
    if usage is not None:
        payload["usage"] = usage
    return httpx.Response(200, json=payload)


def _sse(*events: dict) -> bytes:
    return (
        "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    ).encode()


@pytest.mark.asyncio
@respx.mock
async def test_watsonx_exchanges_iam_token_and_posts_documented_payload():
    token_route = respx.post(IAM_URL).mock(return_value=_token_response())
    api_route = respx.post(DEFAULT_API_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                {"results": [{"generated_text": "Hello"}]},
                {"results": [{"generated_text": " world"}]},
            ),
        )
    )
    llm = WatsonxLLM(_config(), project_id="project-123")

    result = "".join(
        [token async for token in llm.stream("question", temperature=0.6, max_tokens=44)]
    )

    assert result == "Hello world"
    assert token_route.called
    token_form = parse_qs(token_route.calls[0].request.content.decode())
    assert token_form == {
        "grant_type": ["urn:ibm:params:oauth:grant-type:apikey"],
        "apikey": ["ibm-api-key"],
    }
    assert (
        token_route.calls[0]
        .request.headers["content-type"]
        .startswith("application/x-www-form-urlencoded")
    )
    assert api_route.called
    sent = json.loads(api_route.calls[0].request.content)
    assert sent == {
        "model_id": "ibm/granite-3-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "question"},
        ],
        "project_id": "project-123",
        "parameters": {"temperature": 0.6, "max_new_tokens": 44},
    }
    assert api_route.calls[0].request.headers["authorization"] == "Bearer iam-token"
    # Streaming must target the dedicated chat_stream endpoint, not plain chat.
    assert api_route.calls[0].request.url.path == "/ml/v1/text/chat_stream"


@pytest.mark.asyncio
@respx.mock
async def test_watsonx_caches_iam_token_until_expiry_then_refreshes():
    token_route = respx.post(IAM_URL).mock(
        side_effect=[
            _token_response("first"),
            _token_response("second"),
        ]
    )
    api_route = respx.post(DEFAULT_API_URL).mock(
        side_effect=[
            _api_response(results=[{"generated_text": "one"}]),
            _api_response(results=[{"generated_text": "two"}]),
            _api_response(results=[{"generated_text": "three"}]),
        ]
    )
    llm = WatsonxLLM(_config(), project_id="project-123")

    assert await llm.generate("one") == "one"
    assert await llm.generate("two") == "two"
    assert token_route.call_count == 1
    assert api_route.call_count == 2

    llm._token_expires_at = time.monotonic() - 1
    assert await llm.generate("three") == "three"
    assert token_route.call_count == 2
    assert api_route.call_count == 3
    assert api_route.calls[1].request.headers["authorization"] == "Bearer first"
    assert api_route.calls[2].request.headers["authorization"] == "Bearer second"


@pytest.mark.asyncio
@respx.mock
async def test_watsonx_parses_delta_sse_and_usage():
    respx.post(IAM_URL).mock(return_value=_token_response())
    api_route = respx.post(DEFAULT_API_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                {"delta": {"content": "A"}},
                {"results": [{"delta": {"content": "B", "generated_token_count": 2}}]},
                {"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
            ),
        )
    )
    llm = WatsonxLLM(_config(), project_id="project-123")

    assert "".join([token async for token in llm.stream("hi")]) == "AB"
    assert llm.tokens_used == {"input": 7, "output": 3}
    assert api_route.called


@pytest.mark.asyncio
@respx.mock
async def test_watsonx_supports_space_id_custom_endpoint_and_version():
    respx.post(IAM_URL).mock(return_value=_token_response())
    api_route = respx.post(CUSTOM_API_URL).mock(
        return_value=_api_response(
            results=[{"generated_text": "ok", "input_token_count": 2, "generated_token_count": 1}]
        )
    )
    llm = WatsonxLLM(
        _config(),
        space_id="space-456",
        base_url="https://watson.example.test/",
        version="2025-01-15",
    )

    assert await llm.generate("hi") == "ok"
    sent = json.loads(api_route.calls[0].request.content)
    assert sent["space_id"] == "space-456"
    assert "project_id" not in sent
    assert llm.tokens_used == {"input": 2, "output": 1}


@pytest.mark.asyncio
@respx.mock
async def test_watsonx_reads_credentials_from_environment_at_request_time(monkeypatch):
    monkeypatch.setenv("WATSONX_API_KEY", "env-ibm-key")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "env-project")
    respx.post(IAM_URL).mock(return_value=_token_response())
    api_route = respx.post(DEFAULT_API_URL).mock(
        return_value=_api_response(results=[{"generated_text": "ok"}])
    )
    llm = WatsonxLLM(_config(api_key=""))

    assert await llm.generate("hi") == "ok"
    token_request = respx.calls.last
    assert token_request is not None
    assert json.loads(api_route.calls[0].request.content)["project_id"] == "env-project"


@pytest.mark.asyncio
@respx.mock
async def test_watsonx_rejects_missing_credentials_before_network():
    llm = WatsonxLLM(_config(api_key=""))

    with pytest.raises(ValueError, match="project_id or space_id"):
        _ = [token async for token in llm.stream("hi")]

    assert not respx.calls


@pytest.mark.asyncio
@respx.mock
async def test_watsonx_propagates_token_and_api_http_errors():
    respx.post(IAM_URL).mock(return_value=httpx.Response(401, text="bad key"))
    llm = WatsonxLLM(_config(), project_id="project-123")
    with pytest.raises(httpx.HTTPStatusError):
        _ = [token async for token in llm.stream("hi")]

    respx.reset()
    respx.post(IAM_URL).mock(return_value=_token_response())
    respx.post(DEFAULT_API_URL).mock(return_value=httpx.Response(500, text="bad request"))
    with pytest.raises(httpx.HTTPStatusError):
        _ = [token async for token in llm.stream("hi")]


def test_watsonx_stream_is_async_generator_function():
    assert inspect.isasyncgenfunction(WatsonxLLM.stream)
