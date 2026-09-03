"""Native tool-calling contracts for bespoke issue #887 adapters."""

from __future__ import annotations

import json

import httpx
import pytest

respx = pytest.importorskip("respx")

from synapsekit.llm.base import LLMConfig  # noqa: E402
from synapsekit.llm.nvidia_nim import NvidiaNIMLLM  # noqa: E402
from synapsekit.llm.snowflake_cortex import SnowflakeCortexLLM  # noqa: E402
from synapsekit.llm.watsonx import WatsonxLLM  # noqa: E402


def _tool() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _messages() -> list[dict]:
    return [{"role": "user", "content": "What is the weather?"}]


def _config(provider: str, model: str, api_key: str) -> LLMConfig:
    return LLMConfig(model=model, api_key=api_key, provider=provider, max_retries=0)


def _openai_tool_response() -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":"Delhi"}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }


@pytest.mark.asyncio
@respx.mock
async def test_nim_normalizes_native_tool_calls():
    route = respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_tool_response())
    )
    llm = NvidiaNIMLLM(_config("nvidia-nim", "meta/llama", "nim-key"))

    result = await llm.call_with_tools(_messages(), _tool())

    assert result == {
        "content": None,
        "tool_calls": [{"id": "call-1", "name": "weather", "arguments": {"city": "Delhi"}}],
    }
    sent = json.loads(route.calls[0].request.content)
    assert sent["tools"] == _tool()
    assert sent["tool_choice"] == "auto"
    assert llm.tokens_used == {"input": 4, "output": 2}


@pytest.mark.asyncio
@respx.mock
async def test_watsonx_normalizes_native_tool_calls():
    respx.post("https://iam.cloud.ibm.com/identity/token").mock(
        return_value=httpx.Response(200, json={"access_token": "iam-token", "expires_in": 3600})
    )
    route = respx.post("https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2024-10-01").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Delhi"}',
                                },
                            }
                        ]
                    }
                ]
            },
        )
    )
    llm = WatsonxLLM(_config("watsonx", "ibm/granite", "ibm-key"), project_id="project-1")

    result = await llm.call_with_tools(_messages(), _tool())

    assert result["tool_calls"] == [
        {"id": "call-2", "name": "weather", "arguments": {"city": "Delhi"}}
    ]
    sent = json.loads(route.calls[0].request.content)
    assert sent["tools"] == _tool()
    assert sent["tool_choice_option"] == "auto"


@pytest.mark.asyncio
@respx.mock
async def test_watsonx_tool_request_forwards_top_p():
    respx.post("https://iam.cloud.ibm.com/identity/token").mock(
        return_value=httpx.Response(200, json={"access_token": "iam-token", "expires_in": 3600})
    )
    route = respx.post("https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2024-10-01").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-2",
                                    "function": {
                                        "name": "weather",
                                        "arguments": '{"city":"Delhi"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )
    )
    llm = WatsonxLLM(
        LLMConfig(model="ibm/granite", api_key="ibm-key", provider="watsonx", top_p=0.85),
        project_id="project-1",
    )

    result = await llm.call_with_tools(_messages(), _tool())

    assert result["tool_calls"] == [
        {"id": "call-2", "name": "weather", "arguments": {"city": "Delhi"}}
    ]
    sent = json.loads(route.calls[0].request.content)
    assert sent["tool_choice_option"] == "auto"
    assert sent["parameters"]["top_p"] == 0.85


@pytest.mark.asyncio
@respx.mock
async def test_cortex_normalizes_native_tool_calls():
    route = respx.post("https://snowflake.example.test/api/v2/cortex/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_openai_tool_response())
    )
    llm = SnowflakeCortexLLM(
        _config("snowflake-cortex", "mistral-large2", "snowflake-token"),
        base_url="https://snowflake.example.test",
    )

    result = await llm.call_with_tools(_messages(), _tool())

    assert result["tool_calls"] == [
        {"id": "call-1", "name": "weather", "arguments": {"city": "Delhi"}}
    ]
    sent = json.loads(route.calls[0].request.content)
    assert sent["tools"] == _tool()
    assert sent["tool_choice"] == "auto"
    assert sent["max_completion_tokens"] == llm.config.max_tokens
