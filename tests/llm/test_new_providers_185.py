"""Tests for Moonshot, Zhipu, and Cloudflare LLM providers — mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig


def _config(provider: str = "openai", model: str = "test-model") -> LLMConfig:
    return LLMConfig(
        model=model,
        api_key="test-key",
        provider=provider,
        system_prompt="You are helpful.",
        temperature=0.2,
        max_tokens=100,
    )


def _mock_stream_response(texts: list[str]):
    """Create a mock async streaming response with the given text chunks."""
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunks.append(chunk)

    async def async_iter():
        for c in chunks:
            yield c

    mock_response = MagicMock()
    mock_response.__aiter__ = lambda self: async_iter()
    return mock_response


def _mock_tool_response(content: str | None = None, tool_calls: list | None = None):
    """Create a mock non-streaming response for tool calling."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = msg
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    return response


# ------------------------------------------------------------------ #
# MoonshotLLM
# ------------------------------------------------------------------ #


class TestMoonshotLLM:
    def test_construction(self):
        from synapsekit.llm.moonshot import MoonshotLLM

        llm = MoonshotLLM(_config("moonshot", "moonshot-v1-8k"))
        assert llm.config.model == "moonshot-v1-8k"
        assert llm._base_url == "https://api.moonshot.cn/v1"

    def test_custom_base_url(self):
        from synapsekit.llm.moonshot import MoonshotLLM

        llm = MoonshotLLM(_config(), base_url="https://custom.api/v1")
        assert llm._base_url == "https://custom.api/v1"

    def test_import_error_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            from synapsekit.llm.moonshot import MoonshotLLM

            llm = MoonshotLLM(_config("moonshot", "moonshot-v1-8k"))
            llm._client = None
            with pytest.raises(ImportError, match="openai"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        from synapsekit.llm.moonshot import MoonshotLLM

        llm = MoonshotLLM(_config("moonshot", "moonshot-v1-8k"))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_stream_response(["Hello", " world"])
        )
        llm._client = mock_client

        tokens = []
        async for t in llm.stream("hi"):
            tokens.append(t)
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_generate(self):
        from synapsekit.llm.moonshot import MoonshotLLM

        llm = MoonshotLLM(_config("moonshot", "moonshot-v1-8k"))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_stream_response(["Hello", " world"])
        )
        llm._client = mock_client

        result = await llm.generate("hi")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_call_with_tools(self):
        from synapsekit.llm.moonshot import MoonshotLLM

        llm = MoonshotLLM(_config("moonshot", "moonshot-v1-8k"))

        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "calculator"
        tc.function.arguments = json.dumps({"expr": "2+2"})

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_tool_response(tool_calls=[tc])
        )
        llm._client = mock_client

        result = await llm._call_with_tools_impl(
            [{"role": "user", "content": "calc"}],
            [{"type": "function", "function": {"name": "calculator"}}],
        )
        assert result["tool_calls"][0]["name"] == "calculator"
        assert result["tool_calls"][0]["arguments"] == {"expr": "2+2"}


# ------------------------------------------------------------------ #
# ZhipuLLM
# ------------------------------------------------------------------ #


class TestZhipuLLM:
    def test_construction(self):
        from synapsekit.llm.zhipu import ZhipuLLM

        llm = ZhipuLLM(_config("zhipu", "glm-4"))
        assert llm.config.model == "glm-4"
        assert llm._base_url == "https://open.bigmodel.cn/api/paas/v4"

    def test_import_error_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            from synapsekit.llm.zhipu import ZhipuLLM

            llm = ZhipuLLM(_config("zhipu", "glm-4"))
            llm._client = None
            with pytest.raises(ImportError, match="openai"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        from synapsekit.llm.zhipu import ZhipuLLM

        llm = ZhipuLLM(_config("zhipu", "glm-4"))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_stream_response(["Zhipu", " rocks"])
        )
        llm._client = mock_client

        tokens = []
        async for t in llm.stream("hi"):
            tokens.append(t)
        assert tokens == ["Zhipu", " rocks"]

    @pytest.mark.asyncio
    async def test_generate(self):
        from synapsekit.llm.zhipu import ZhipuLLM

        llm = ZhipuLLM(_config("zhipu", "glm-4"))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_stream_response(["GLM", " answer"])
        )
        llm._client = mock_client

        result = await llm.generate("hi")
        assert result == "GLM answer"

    @pytest.mark.asyncio
    async def test_call_with_tools(self):
        from synapsekit.llm.zhipu import ZhipuLLM

        llm = ZhipuLLM(_config("zhipu", "glm-4"))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_tool_response(content="No tools needed")
        )
        llm._client = mock_client

        result = await llm._call_with_tools_impl(
            [{"role": "user", "content": "hello"}],
            [{"type": "function", "function": {"name": "calc"}}],
        )
        assert result["content"] == "No tools needed"
        assert result["tool_calls"] is None


# ------------------------------------------------------------------ #
# CloudflareLLM (based on tests by @adaumsilva, PR #320)
# ------------------------------------------------------------------ #


async def _line_iter(*lines: str):
    for line in lines:
        yield line


def _mock_httpx_stream(lines: list[str]) -> MagicMock:
    """Return a mocked httpx client whose .stream() is a usable async CM."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = lambda: _line_iter(*lines)

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_cm
    return mock_client


class TestCloudflareLLM:
    @pytest.fixture
    def llm(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        return CloudflareLLM(
            _config("cloudflare", "@cf/meta/llama-3.1-8b-instruct"), account_id="test-account"
        )

    def test_endpoint_format(self, llm):
        assert llm._endpoint() == (
            "https://api.cloudflare.com/client/v4/accounts"
            "/test-account/ai/run/@cf/meta/llama-3.1-8b-instruct"
        )

    def test_custom_base_url(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        llm = CloudflareLLM(_config(), account_id="acct", base_url="https://custom.example.com")
        assert llm._endpoint().startswith("https://custom.example.com")

    def test_raises_without_account_id(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        llm = CloudflareLLM(_config("cloudflare", "@cf/model"))
        with pytest.raises(ValueError, match="account_id"):
            llm._get_client()

    def test_import_error_without_httpx(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        llm = CloudflareLLM(_config(), account_id="abc")
        with patch.dict("sys.modules", {"httpx": None}):
            llm._client = None
            with pytest.raises(ImportError, match="httpx"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, llm):
        llm._client = _mock_httpx_stream(
            [
                'data: {"response": "Hello"}',
                'data: {"response": " world"}',
                "data: [DONE]",
            ]
        )
        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_skips_non_data_lines(self, llm):
        llm._client = _mock_httpx_stream(
            [
                ": ping",
                "",
                'data: {"response": "Token"}',
                "data: [DONE]",
            ]
        )
        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Token"]

    @pytest.mark.asyncio
    async def test_stream_skips_invalid_json(self, llm):
        llm._client = _mock_httpx_stream(
            [
                "data: not-json",
                'data: {"response": "Good"}',
                "data: [DONE]",
            ]
        )
        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["Good"]

    @pytest.mark.asyncio
    async def test_stream_stops_at_done(self, llm):
        llm._client = _mock_httpx_stream(
            [
                'data: {"response": "A"}',
                "data: [DONE]",
                'data: {"response": "B"}',
            ]
        )
        tokens = [t async for t in llm.stream("hi")]
        assert tokens == ["A"]

    @pytest.mark.asyncio
    async def test_generate_collects_tokens(self, llm):
        llm._client = _mock_httpx_stream(
            [
                'data: {"response": "Hello"}',
                'data: {"response": " world"}',
                "data: [DONE]",
            ]
        )
        result = await llm.generate("hi")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_output_tokens_tracked(self, llm):
        llm._client = _mock_httpx_stream(
            [
                'data: {"response": "A"}',
                'data: {"response": "B"}',
                'data: {"response": "C"}',
                "data: [DONE]",
            ]
        )
        _ = [t async for t in llm.stream("hi")]
        assert llm._output_tokens == 3

    @pytest.mark.asyncio
    async def test_stream_with_messages_passes_kwargs(self, llm):
        llm._client = _mock_httpx_stream(['data: {"response": "x"}', "data: [DONE]"])
        messages = [{"role": "user", "content": "q"}]
        _ = [t async for t in llm.stream_with_messages(messages, temperature=0.9, max_tokens=512)]
        payload = llm._client.stream.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 512


# ------------------------------------------------------------------ #
# Facade auto-detection
# ------------------------------------------------------------------ #


class TestFacadeAutoDetection:
    def test_moonshot_detection(self):
        from synapsekit.rag.facade import _make_llm

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from synapsekit.llm.moonshot import MoonshotLLM

            llm = _make_llm("moonshot-v1-8k", "key", None, "sys", 0.2, 100)
            assert isinstance(llm, MoonshotLLM)

    def test_zhipu_detection(self):
        from synapsekit.rag.facade import _make_llm

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from synapsekit.llm.zhipu import ZhipuLLM

            llm = _make_llm("glm-4", "key", None, "sys", 0.2, 100)
            assert isinstance(llm, ZhipuLLM)

    def test_cloudflare_detection(self):
        from synapsekit.rag.facade import _make_llm

        with patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "abc123"}):
            from synapsekit.llm.cloudflare import CloudflareLLM

            llm = _make_llm("@cf/meta/llama-3-8b-instruct", "key", None, "sys", 0.2, 100)
            assert isinstance(llm, CloudflareLLM)
