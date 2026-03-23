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
# CloudflareLLM
# ------------------------------------------------------------------ #


class TestCloudflareLLM:
    def test_construction_with_account_id(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        llm = CloudflareLLM(
            _config("cloudflare", "@cf/meta/llama-3-8b-instruct"), account_id="abc123"
        )
        assert llm._account_id == "abc123"

    def test_construction_with_base_url(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        llm = CloudflareLLM(_config(), base_url="https://custom.workers.ai/v1")
        assert llm._base_url == "https://custom.workers.ai/v1"

    def test_raises_without_account_id_or_base_url(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        llm = CloudflareLLM(_config("cloudflare", "@cf/model"))
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            llm._client = None
            with pytest.raises(ValueError, match="account_id or base_url"):
                llm._get_client()

    def test_import_error_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            from synapsekit.llm.cloudflare import CloudflareLLM

            llm = CloudflareLLM(_config(), account_id="abc")
            llm._client = None
            with pytest.raises(ImportError, match="openai"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        llm = CloudflareLLM(_config("cloudflare", "@cf/model"), account_id="abc")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_stream_response(["Cloud", "flare"])
        )
        llm._client = mock_client

        tokens = []
        async for t in llm.stream("hi"):
            tokens.append(t)
        assert tokens == ["Cloud", "flare"]

    @pytest.mark.asyncio
    async def test_generate(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        llm = CloudflareLLM(_config(), base_url="https://test.api/v1")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_stream_response(["Hi ", "there"])
        )
        llm._client = mock_client

        result = await llm.generate("hello")
        assert result == "Hi there"

    @pytest.mark.asyncio
    async def test_call_with_tools(self):
        from synapsekit.llm.cloudflare import CloudflareLLM

        llm = CloudflareLLM(_config(), account_id="abc")

        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "search"
        tc.function.arguments = json.dumps({"query": "test"})

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_tool_response(tool_calls=[tc])
        )
        llm._client = mock_client

        result = await llm._call_with_tools_impl(
            [{"role": "user", "content": "search something"}],
            [{"type": "function", "function": {"name": "search"}}],
        )
        assert result["tool_calls"][0]["name"] == "search"
        assert result["tool_calls"][0]["arguments"] == {"query": "test"}
        assert llm._input_tokens == 10
        assert llm._output_tokens == 5


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

        mock_openai = MagicMock()
        with (
            patch.dict("sys.modules", {"openai": mock_openai}),
            patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "abc123"}),
        ):
            from synapsekit.llm.cloudflare import CloudflareLLM

            llm = _make_llm("@cf/meta/llama-3-8b-instruct", "key", None, "sys", 0.2, 100)
            assert isinstance(llm, CloudflareLLM)
