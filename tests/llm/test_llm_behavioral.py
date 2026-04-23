"""Behavioral tests for LLM providers.

Verifies that generate(), stream(), stream_with_messages(), call_with_tools(),
token counting, and ImportError handling behave correctly for all major providers.
All network calls are mocked — no real API keys needed.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(provider: str = "openai", model: str = "gpt-4o-mini", **kw) -> LLMConfig:
    return LLMConfig(model=model, api_key="sk-test", provider=provider, **kw)


async def _async_gen(*tokens: str) -> AsyncGenerator[str, None]:
    for t in tokens:
        yield t


# ---------------------------------------------------------------------------
# 1. BaseLLM.generate() — collects stream tokens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_generate_collects_stream():
    """generate() joins all streamed tokens into one string."""
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg())

    async def fake_stream(prompt, **kw):
        for tok in ["Hello", ", ", "world", "!"]:
            yield tok

    with patch.object(llm, "stream", fake_stream):
        result = await llm.generate("hi")

    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_base_generate_with_messages_collects_stream():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg())

    async def fake_stream_msgs(messages, **kw):
        yield "answer"

    with patch.object(llm, "stream_with_messages", fake_stream_msgs):
        result = await llm.generate_with_messages([{"role": "user", "content": "q"}])

    assert result == "answer"


# ---------------------------------------------------------------------------
# 2. OpenAILLM — stream + tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_yields_tokens():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg())

    # Mock AsyncOpenAI client
    chunk1 = MagicMock()
    chunk1.usage = None
    chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]

    chunk2 = MagicMock()
    chunk2.usage = MagicMock(prompt_tokens=5, completion_tokens=3)
    chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]

    mock_stream = MagicMock()

    async def _aiter(self):
        for c in [chunk1, chunk2]:
            yield c

    mock_stream.__aiter__ = _aiter

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    llm._client = mock_client

    tokens = [t async for t in llm.stream("hi")]
    assert tokens == ["Hello", " world"]
    assert llm.tokens_used["input"] == 5
    assert llm.tokens_used["output"] == 3


@pytest.mark.asyncio
async def test_openai_call_with_tools_returns_tool_calls():
    import json

    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg())

    tc = MagicMock()
    tc.id = "call_abc123"
    tc.function.name = "get_weather"
    tc.function.arguments = json.dumps({"city": "NYC"})

    mock_msg = MagicMock()
    mock_msg.tool_calls = [tc]
    mock_msg.content = None

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=mock_msg)]
    mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
    llm._client = mock_client

    result = await llm.call_with_tools(
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )

    assert result["tool_calls"] is not None
    assert result["tool_calls"][0]["name"] == "get_weather"
    assert result["tool_calls"][0]["arguments"] == {"city": "NYC"}
    assert result["content"] is None


@pytest.mark.asyncio
async def test_openai_call_with_tools_returns_text_when_no_calls():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg())
    mock_msg = MagicMock()
    mock_msg.tool_calls = None
    mock_msg.content = "I can help with that."

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=mock_msg)]
    mock_resp.usage = MagicMock(prompt_tokens=3, completion_tokens=7)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
    llm._client = mock_client

    result = await llm.call_with_tools(
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
    )
    assert result["content"] == "I can help with that."
    assert result["tool_calls"] is None


def test_openai_get_client_import_error():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg())
    with patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(ImportError, match="openai package required"):
            llm._get_client()


# ---------------------------------------------------------------------------
# 3. AnthropicLLM — stream + tool use
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_stream_yields_tokens():
    from synapsekit.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM(_cfg(provider="anthropic", model="claude-haiku-4-5-20251001"))

    mock_stream_ctx = MagicMock()

    async def _text_stream():
        for t in ["Hi", " there"]:
            yield t

    mock_stream_ctx.text_stream = _text_stream()
    final_msg = MagicMock()
    final_msg.usage = MagicMock(input_tokens=4, output_tokens=2)
    mock_stream_ctx.get_final_message = AsyncMock(return_value=final_msg)
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(return_value=mock_stream_ctx)
    llm._client = mock_client

    tokens = [t async for t in llm.stream("hey")]
    assert tokens == ["Hi", " there"]
    assert llm.tokens_used["input"] == 4
    assert llm.tokens_used["output"] == 2


@pytest.mark.asyncio
async def test_anthropic_stream_with_messages_extracts_system():
    """stream_with_messages strips system message and passes it separately."""
    from synapsekit.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM(_cfg(provider="anthropic", model="claude-haiku-4-5-20251001"))

    captured_calls = {}

    mock_stream_ctx = MagicMock()

    async def _text_stream():
        yield "ok"

    mock_stream_ctx.text_stream = _text_stream()
    final_msg = MagicMock()
    final_msg.usage = MagicMock(input_tokens=1, output_tokens=1)
    mock_stream_ctx.get_final_message = AsyncMock(return_value=final_msg)
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    def _capture_stream(**kw):
        captured_calls.update(kw)
        return mock_stream_ctx

    mock_client = MagicMock()
    mock_client.messages.stream = MagicMock(side_effect=lambda **kw: _capture_stream(**kw))
    llm._client = mock_client

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    [t async for t in llm.stream_with_messages(messages)]

    assert captured_calls["system"] == "You are helpful."
    user_msgs = [m for m in captured_calls["messages"] if m.get("role") == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_anthropic_call_with_tools_returns_tool_use():
    from synapsekit.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM(_cfg(provider="anthropic", model="claude-haiku-4-5-20251001"))

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "tu_001"
    tool_use_block.name = "search"
    tool_use_block.input = {"query": "Python docs"}

    mock_resp = MagicMock()
    mock_resp.content = [tool_use_block]
    mock_resp.usage = MagicMock(input_tokens=8, output_tokens=4)

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_resp)
    llm._client = mock_client

    result = await llm.call_with_tools(
        messages=[{"role": "user", "content": "search Python"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Web search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    assert result["tool_calls"][0]["name"] == "search"
    assert result["tool_calls"][0]["arguments"] == {"query": "Python docs"}


@pytest.mark.asyncio
async def test_anthropic_call_with_tools_converts_tool_message():
    """tool-result messages are converted from OpenAI format to Anthropic format."""
    from synapsekit.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM(_cfg(provider="anthropic", model="claude-haiku-4-5-20251001"))

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "The result is 42."

    mock_resp = MagicMock()
    mock_resp.content = [text_block]
    mock_resp.usage = MagicMock(input_tokens=5, output_tokens=5)

    captured = {}

    async def _create(**kw):
        captured.update(kw)
        return mock_resp

    mock_client = MagicMock()
    mock_client.messages.create = _create
    llm._client = mock_client

    messages = [
        {"role": "user", "content": "What is 6*7?"},
        {"role": "tool", "tool_call_id": "call_001", "content": "42"},
    ]
    await llm.call_with_tools(messages=messages, tools=[])

    # Verify tool result was converted to Anthropic's tool_result format
    converted = captured["messages"]
    tool_result_msg = next((m for m in converted if isinstance(m.get("content"), list)), None)
    assert tool_result_msg is not None
    assert tool_result_msg["content"][0]["type"] == "tool_result"


def test_anthropic_get_client_import_error():
    from synapsekit.llm.anthropic import AnthropicLLM

    llm = AnthropicLLM(_cfg(provider="anthropic", model="claude-haiku-4-5-20251001"))
    with patch.dict(sys.modules, {"anthropic": None}):
        with pytest.raises(ImportError, match="anthropic package required"):
            llm._get_client()


# ---------------------------------------------------------------------------
# 4. GeminiLLM — stream + import guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_stream_yields_tokens():
    from synapsekit.llm.gemini import GeminiLLM

    llm = GeminiLLM(_cfg(provider="gemini", model="gemini-1.5-flash"))

    chunk1 = MagicMock()
    chunk1.text = "Gemini"
    chunk2 = MagicMock()
    chunk2.text = " response"

    async def _fake_generate_async(prompt, generation_config, stream):
        async def _chunks():
            for c in [chunk1, chunk2]:
                yield c

        return _chunks()

    mock_model = MagicMock()
    mock_model.generate_content_async = _fake_generate_async
    llm._model = mock_model

    tokens = [t async for t in llm.stream("hello")]
    assert "".join(tokens) == "Gemini response"
    assert llm._output_tokens == 2


def test_gemini_get_model_import_error():
    from synapsekit.llm.gemini import GeminiLLM

    llm = GeminiLLM(_cfg(provider="gemini", model="gemini-1.5-flash"))
    with patch.dict(sys.modules, {"google": None, "google.generativeai": None}):
        with pytest.raises(ImportError, match="google-generativeai required"):
            llm._get_model()


@pytest.mark.asyncio
async def test_gemini_call_with_tools_returns_tool_calls():
    from synapsekit.llm.gemini import GeminiLLM

    llm = GeminiLLM(_cfg(provider="gemini", model="gemini-1.5-flash"))

    part = MagicMock()
    part.function_call = MagicMock()
    part.function_call.name = "get_time"
    part.function_call.args = {"tz": "UTC"}
    part.text = None

    candidate = MagicMock()
    candidate.content.parts = [part]

    mock_resp = MagicMock()
    mock_resp.candidates = [candidate]

    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_resp)
    llm._model = mock_model

    # patch genai import inside _call_with_tools_impl
    mock_genai = MagicMock()
    mock_genai.protos.FunctionDeclaration = MagicMock(return_value=MagicMock())
    mock_genai.protos.Tool = MagicMock(return_value=MagicMock())

    mock_google = MagicMock()
    mock_google.generativeai = mock_genai
    with patch.dict(sys.modules, {"google": mock_google, "google.generativeai": mock_genai}):
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "What time is it?"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_time", "description": "Get time", "parameters": {}},
                }
            ],
        )

    assert result["tool_calls"] is not None
    assert result["tool_calls"][0]["name"] == "get_time"


# ---------------------------------------------------------------------------
# 5. GroqLLM — stream + tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_stream_yields_tokens():
    from synapsekit.llm.groq import GroqLLM

    llm = GroqLLM(_cfg(provider="groq", model="llama-3.3-70b-versatile"))

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Fast"))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content=" answer"))]
    chunk3 = MagicMock()
    chunk3.choices = [MagicMock(delta=MagicMock(content=None))]  # empty delta

    mock_stream = MagicMock()

    async def _aiter(self):
        for c in [chunk1, chunk2, chunk3]:
            yield c

    mock_stream.__aiter__ = _aiter

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    llm._client = mock_client

    tokens = [t async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}])]
    assert tokens == ["Fast", " answer"]  # None delta skipped
    assert llm._output_tokens == 2


def test_groq_get_client_import_error():
    from synapsekit.llm.groq import GroqLLM

    llm = GroqLLM(_cfg(provider="groq", model="llama-3.3-70b-versatile"))
    with patch.dict(sys.modules, {"groq": None}):
        with pytest.raises(ImportError, match="groq package required"):
            llm._get_client()


# ---------------------------------------------------------------------------
# 6. OllamaLLM — stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_stream_yields_tokens():
    from synapsekit.llm.ollama import OllamaLLM

    llm = OllamaLLM(_cfg(provider="ollama", model="llama3"))

    async def _fake_chat(model, messages, stream, options):
        for content in ["Local", " model"]:
            yield {"message": {"content": content}}

    mock_client = MagicMock()
    mock_client.chat = _fake_chat
    llm._client = mock_client

    tokens = [t async for t in llm.stream("hello")]
    assert "".join(tokens) == "Local model"


def test_ollama_get_client_import_error():
    from synapsekit.llm.ollama import OllamaLLM

    llm = OllamaLLM(_cfg(provider="ollama", model="llama3"))
    with patch.dict(sys.modules, {"ollama": None}):
        with pytest.raises(ImportError, match="ollama required"):
            llm._get_client()


# ---------------------------------------------------------------------------
# 7. BaseLLM token counting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_counting_accumulates():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg())

    chunk = MagicMock()
    chunk.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    chunk.choices = [MagicMock(delta=MagicMock(content="test"))]

    mock_stream = MagicMock()

    async def _aiter(self):
        yield chunk

    mock_stream.__aiter__ = _aiter

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
    llm._client = mock_client

    await llm.generate("prompt1")
    assert llm.tokens_used["input"] == 10
    assert llm.tokens_used["output"] == 5

    # Second generate accumulates
    await llm.generate("prompt2")
    assert llm.tokens_used["input"] == 20
    assert llm.tokens_used["output"] == 10


@pytest.mark.asyncio
async def test_token_reset():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg())
    llm._input_tokens = 50
    llm._output_tokens = 30
    llm._reset_tokens()
    assert llm.tokens_used == {"input": 0, "output": 0}


# ---------------------------------------------------------------------------
# 8. BaseLLM.generate() with caching (memory backend)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_caches_result():
    """Second identical generate() call returns cached result without hitting stream."""
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg(cache=True, cache_backend="memory"))

    call_count = 0

    async def counting_stream(prompt, **kw):
        nonlocal call_count
        call_count += 1
        yield "cached answer"

    with patch.object(llm, "stream", counting_stream):
        r1 = await llm.generate("same prompt")
        r2 = await llm.generate("same prompt")

    assert r1 == r2 == "cached answer"
    assert call_count == 1  # stream only called once
    assert llm.cache_stats["hits"] == 1
    assert llm.cache_stats["misses"] == 1


@pytest.mark.asyncio
async def test_generate_cache_miss_on_different_prompt():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg(cache=True, cache_backend="memory"))
    call_count = 0

    async def counting_stream(prompt, **kw):
        nonlocal call_count
        call_count += 1
        yield f"answer to: {prompt}"

    with patch.object(llm, "stream", counting_stream):
        r1 = await llm.generate("prompt A")
        r2 = await llm.generate("prompt B")

    assert r1 != r2
    assert call_count == 2
    assert llm.cache_stats["misses"] == 2


@pytest.mark.asyncio
async def test_generate_no_cache_when_disabled():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg(cache=False))

    async def counting_stream(prompt, **kw):
        yield "result"

    with patch.object(llm, "stream", counting_stream):
        await llm.generate("same")
        await llm.generate("same")

    assert llm.cache_stats == {}  # no cache object


# ---------------------------------------------------------------------------
# 9. cache_stats property
# ---------------------------------------------------------------------------


def test_cache_stats_empty_when_no_cache():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg(cache=False))
    assert llm.cache_stats == {}


def test_cache_stats_present_when_cache_enabled():
    from synapsekit.llm.openai import OpenAILLM

    llm = OpenAILLM(_cfg(cache=True, cache_backend="memory"))
    stats = llm.cache_stats
    assert "hits" in stats
    assert "misses" in stats
    assert "size" in stats
    assert stats["hits"] == 0


# ---------------------------------------------------------------------------
# 10. _call_with_tools_impl raises NotImplementedError for base providers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_providers_without_native_tool_calling_raise():
    """OllamaLLM inherits BaseLLM._call_with_tools_impl which raises NotImplementedError."""
    from synapsekit.llm.ollama import OllamaLLM

    llm = OllamaLLM(_cfg(provider="ollama", model="llama3"))
    with pytest.raises(NotImplementedError):
        await llm._call_with_tools_impl([], [])
