from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.llm.edge import EdgeFallbackBlockedError, EdgeRuntime, FallbackPolicy


class FakeLLM(BaseLLM):
    def __init__(
        self,
        *,
        name: str,
        response: str,
        supports_tools: bool = False,
        fail_generate: bool = False,
    ) -> None:
        super().__init__(LLMConfig(model=name, api_key="", provider=name))
        self.response = response
        self.supports_tools = supports_tools
        self.fail_generate = fail_generate
        self.prompts: list[str] = []
        self.messages: list[list[dict[str, Any]]] = []

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        if self.fail_generate:
            raise RuntimeError("local failed")
        self.prompts.append(prompt)
        for token in self.response.split("|"):
            yield token

    async def generate(self, prompt: str, **kw: Any) -> str:
        if self.fail_generate:
            raise RuntimeError("local failed")
        self.prompts.append(prompt)
        return self.response.replace("|", "")

    async def generate_with_messages(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        self.messages.append(messages)
        return self.response.replace("|", "")

    async def stream_with_messages(
        self,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> AsyncGenerator[str]:
        self.messages.append(messages)
        for token in self.response.split("|"):
            yield token

    async def _call_with_tools_impl(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.supports_tools:
            raise NotImplementedError("tools unsupported")
        self.messages.append(messages)
        return {"content": self.response, "tool_calls": None}


@pytest.mark.asyncio
async def test_local_generate_is_default() -> None:
    runtime = EdgeRuntime(local_llm=FakeLLM(name="local", response="ok"))

    result = await runtime.generate("hello")

    assert result == "ok"
    assert runtime.last_route == "local"
    assert runtime.last_fallback_reason is None


@pytest.mark.asyncio
async def test_context_overflow_routes_to_cloud() -> None:
    local = FakeLLM(name="local", response="local")
    cloud = FakeLLM(name="cloud", response="cloud")
    runtime = EdgeRuntime(
        local_llm=local,
        cloud_llm=cloud,
        fallback=FallbackPolicy(if_context_exceeds=1),
    )

    result = await runtime.generate("this is definitely longer than one token")

    assert result == "cloud"
    assert runtime.last_route == "cloud"
    assert runtime.last_fallback_reason == "context_exceeds"
    assert local.prompts == []
    assert cloud.prompts


@pytest.mark.asyncio
async def test_cloud_fallback_requires_cloud_model() -> None:
    runtime = EdgeRuntime(
        local_llm=FakeLLM(name="local", response="local"),
        fallback=FallbackPolicy(if_context_exceeds=1),
    )

    with pytest.raises(EdgeFallbackBlockedError, match="no cloud_llm"):
        await runtime.generate("this prompt exceeds the policy threshold")


@pytest.mark.asyncio
async def test_user_opt_in_gate() -> None:
    cloud = FakeLLM(name="cloud", response="cloud")
    runtime = EdgeRuntime(
        local_llm=FakeLLM(name="local", response="local"),
        cloud_llm=cloud,
        fallback=FallbackPolicy(if_user_opts_in=True),
    )

    assert await runtime.generate("hello") == "local"
    assert await runtime.generate("hello", allow_cloud_fallback=True) == "cloud"
    assert runtime.last_fallback_reason == "user_opt_in"


@pytest.mark.asyncio
async def test_pii_is_redacted_before_cloud_generate() -> None:
    cloud = FakeLLM(name="cloud", response="cloud")
    runtime = EdgeRuntime(
        local_llm=FakeLLM(name="local", response="local"),
        cloud_llm=cloud,
        fallback=FallbackPolicy(if_user_opts_in=True, require_pii_redaction_before_fallback=True),
    )

    await runtime.generate(
        "Email alice@example.com or call 555-123-4567",
        allow_cloud_fallback=True,
    )

    sent = cloud.prompts[-1]
    assert "alice@example.com" not in sent
    assert "555-123-4567" not in sent
    assert "[EMAIL_1]" in sent
    assert "[PHONE_1]" in sent
    assert runtime.last_redaction is not None
    assert runtime.last_metadata.pii_types_found == ("email", "phone")


@pytest.mark.asyncio
async def test_pii_is_redacted_before_cloud_messages() -> None:
    cloud = FakeLLM(name="cloud", response="cloud")
    runtime = EdgeRuntime(
        local_llm=FakeLLM(name="local", response="local"),
        cloud_llm=cloud,
        fallback=FallbackPolicy(if_user_opts_in=True),
    )

    await runtime.generate_with_messages(
        [{"role": "user", "content": "My SSN is 123-45-6789"}],
        allow_cloud_fallback=True,
    )

    assert cloud.messages[-1][0]["content"] == "My SSN is [SSN_1]"
    assert runtime.last_metadata.pii_types_found == ("ssn",)


@pytest.mark.asyncio
async def test_tool_unsupported_falls_back_when_allowed() -> None:
    cloud = FakeLLM(name="cloud", response="tool", supports_tools=True)
    runtime = EdgeRuntime(
        local_llm=FakeLLM(name="local", response="local", supports_tools=False),
        cloud_llm=cloud,
        fallback=FallbackPolicy(if_tool_unsupported_locally=True),
    )

    result = await runtime.call_with_tools(
        [{"role": "user", "content": "use a tool"}],
        [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
    )

    assert result == {"content": "tool", "tool_calls": None}
    assert runtime.last_route == "cloud"
    assert runtime.last_fallback_reason == "tool_unsupported"


@pytest.mark.asyncio
async def test_tool_unsupported_is_blocked_by_default() -> None:
    runtime = EdgeRuntime(
        local_llm=FakeLLM(name="local", response="local", supports_tools=False),
        cloud_llm=FakeLLM(name="cloud", response="cloud", supports_tools=True),
    )

    with pytest.raises(EdgeFallbackBlockedError, match="if_tool_unsupported_locally"):
        await runtime.call_with_tools([{"role": "user", "content": "tool"}], [])


@pytest.mark.asyncio
async def test_stream_cloud_preserves_token_order() -> None:
    runtime = EdgeRuntime(
        local_llm=FakeLLM(name="local", response="local"),
        cloud_llm=FakeLLM(name="cloud", response="a|b|c"),
        fallback=FallbackPolicy(if_user_opts_in=True),
    )

    tokens = [token async for token in runtime.stream("hello", allow_cloud_fallback=True)]

    assert tokens == ["a", "b", "c"]
    assert runtime.last_route == "cloud"


@pytest.mark.asyncio
async def test_local_error_fallback_can_be_enabled() -> None:
    runtime = EdgeRuntime(
        local_llm=FakeLLM(name="local", response="local", fail_generate=True),
        cloud_llm=FakeLLM(name="cloud", response="cloud"),
        fallback=FallbackPolicy(fallback_on_local_error=True),
    )

    assert await runtime.generate("hello") == "cloud"
    assert runtime.last_fallback_reason == "local_error"
