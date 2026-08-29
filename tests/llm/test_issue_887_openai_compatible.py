"""HTTP contracts for the OpenAI-compatible providers from issue #887."""

from __future__ import annotations

import importlib
import inspect
import json
from typing import Any

import httpx
import pytest

from synapsekit.llm.base import LLMConfig

respx = pytest.importorskip("respx")
pytest.importorskip("openai")


PROVIDERS = [
    ("deepinfra", "DeepInfraLLM", "https://api.deepinfra.com/v1/openai", "deepinfra-model"),
    ("nebius", "NebiusLLM", "https://api.tokenfactory.nebius.com/v1", "nebius-model"),
    ("baseten", "BasetenLLM", "https://inference.baseten.co/v1", "baseten-model"),
    ("upstage", "UpstageLLM", "https://api.upstage.ai/v1", "solar-pro"),
    ("reka", "RekaLLM", "https://api.reka.ai/v1", "reka-core"),
    ("hyperbolic", "HyperbolicLLM", "https://api.hyperbolic.xyz/v1", "hyperbolic-model"),
    ("friendli", "FriendliLLM", "https://api.friendli.ai/serverless/v1", "friendli-model"),
    ("clarifai", "ClarifaiLLM", "https://api.clarifai.com/v2/ext/openai/v1", "clarifai-model"),
]


def _provider_class(module_name: str, class_name: str) -> type:
    """Load a provider while turning a missing implementation into a RED failure."""
    try:
        module = importlib.import_module(f"synapsekit.llm.{module_name}")
    except ModuleNotFoundError as exc:
        pytest.fail(f"provider module {module_name!r} is missing: {exc}")
    return getattr(module, class_name)


def _sse(*chunks: dict[str, Any]) -> bytes:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + "data: [DONE]\n\n").encode()


def _chunk(
    delta: dict[str, Any] | None = None, usage: dict[str, int] | None = None
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": "chatcmpl-issue-887",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test-model",
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _config(provider: str, model: str) -> LLMConfig:
    return LLMConfig(
        model=model,
        api_key="issue-887-test-key",
        provider=provider,
        system_prompt="Be concise.",
        temperature=0.2,
        max_tokens=100,
        max_retries=0,
    )


def _json_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-tool-887",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"London"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name,class_name,base_url,model", PROVIDERS)
@respx.mock
async def test_each_provider_streams_sse_and_tracks_usage(
    module_name: str, class_name: str, base_url: str, model: str
) -> None:
    route = respx.post(f"{base_url}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                _chunk({"content": "Hello"}),
                _chunk({"content": " world"}),
                _chunk(usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
            ),
        )
    )
    provider_class = _provider_class(module_name, class_name)
    llm = provider_class(_config(module_name, model))

    output = "".join(
        [token async for token in llm.stream("question", temperature=0.7, max_tokens=50, top_p=0.9)]
    )

    assert output == "Hello world"
    assert route.called
    assert route.calls[0].request.headers["authorization"] == "Bearer issue-887-test-key"
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == model
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["temperature"] == 0.7
    assert sent["top_p"] == 0.9
    assert sent["max_tokens"] == 50
    assert sent["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "question"},
    ]
    assert llm.tokens_used == {"input": 5, "output": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name,class_name,_default_url,model", PROVIDERS)
@respx.mock
async def test_each_provider_honors_custom_base_url(
    module_name: str, class_name: str, _default_url: str, model: str
) -> None:
    custom_base_url = f"https://custom-{module_name}.example/v1"
    route = respx.post(f"{custom_base_url}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(_chunk({"content": "ok"})),
        )
    )
    provider_class = _provider_class(module_name, class_name)
    llm = provider_class(_config(module_name, model), base_url=custom_base_url)

    output = "".join([token async for token in llm.stream("hello")])

    assert output == "ok"
    assert route.called
    assert llm._base_url == custom_base_url


@pytest.mark.asyncio
@pytest.mark.parametrize("module_name,class_name,base_url,model", PROVIDERS)
@respx.mock
async def test_openai_tool_response_is_normalized_for_each_adapter(
    module_name: str, class_name: str, base_url: str, model: str
) -> None:
    route = respx.post(f"{base_url}/chat/completions").mock(
        return_value=httpx.Response(200, json=_json_response())
    )
    provider_class = _provider_class(module_name, class_name)
    llm = provider_class(_config(module_name, model))
    messages = [{"role": "user", "content": "Weather in London?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object"},
            },
        }
    ]

    result = await llm._call_with_tools_impl(messages, tools)

    assert route.called
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == model
    assert sent["messages"] == messages
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"
    assert result == {
        "content": None,
        "tool_calls": [
            {
                "id": "call_weather",
                "name": "get_weather",
                "arguments": {"city": "London"},
            }
        ],
    }
    assert llm.tokens_used == {"input": 11, "output": 7}


@pytest.mark.asyncio
@respx.mock
async def test_reka_preserves_multimodal_message_content() -> None:
    base_url = "https://api.reka.ai/v1"
    route = respx.post(f"{base_url}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(_chunk({"content": "seen"})),
        )
    )
    reka_class = _provider_class("reka", "RekaLLM")
    llm = reka_class(_config("reka", "reka-core"))
    multimodal_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
        }
    ]

    output = "".join([token async for token in llm.stream_with_messages(multimodal_messages)])

    assert output == "seen"
    assert json.loads(route.calls[0].request.content)["messages"] == multimodal_messages


@pytest.mark.parametrize("module_name,class_name,_base_url,model", PROVIDERS)
def test_provider_stream_is_an_async_generator(
    module_name: str, class_name: str, _base_url: str, model: str
) -> None:
    provider_class = _provider_class(module_name, class_name)
    assert inspect.isasyncgenfunction(provider_class.stream)
    assert inspect.isasyncgenfunction(provider_class.stream_with_messages)
