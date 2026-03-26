"""Tests for new LLM providers (Ollama, Cohere, Mistral, Gemini, Bedrock) — mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig


def make_config(provider="openai", model="test-model"):
    return LLMConfig(
        model=model,
        api_key="test-key",
        provider=provider,
        system_prompt="You are helpful.",
        temperature=0.2,
        max_tokens=100,
    )


# ------------------------------------------------------------------ #
# OllamaLLM
# ------------------------------------------------------------------ #


class TestOllamaLLM:
    def test_import_error_without_ollama(self):
        with patch.dict("sys.modules", {"ollama": None}):
            from synapsekit.llm.ollama import OllamaLLM

            llm = OllamaLLM(make_config("ollama", "llama3"))
            llm._client = None  # force re-init
            with pytest.raises(ImportError, match="ollama"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        chunks = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " world"}},
        ]

        async def mock_chat(**kw):
            for c in chunks:
                yield c

        mock_async_client = MagicMock()
        mock_async_client.chat = mock_chat
        mock_ollama = MagicMock()
        mock_ollama.AsyncClient.return_value = mock_async_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from synapsekit.llm.ollama import OllamaLLM

            llm = OllamaLLM(make_config("ollama", "llama3"))
            tokens = []
            async for t in llm.stream("hi"):
                tokens.append(t)
            assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_with_messages(self):
        chunks = [{"message": {"content": "ok"}}]

        async def mock_chat(**kw):
            for c in chunks:
                yield c

        mock_async_client = MagicMock()
        mock_async_client.chat = mock_chat
        mock_ollama = MagicMock()
        mock_ollama.AsyncClient.return_value = mock_async_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from synapsekit.llm.ollama import OllamaLLM

            llm = OllamaLLM(make_config("ollama"))
            tokens = []
            async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}]):
                tokens.append(t)
            assert tokens == ["ok"]


# ------------------------------------------------------------------ #
# CohereLLM
# ------------------------------------------------------------------ #


class TestCohereLLM:
    def test_import_error_without_cohere(self):
        with patch.dict("sys.modules", {"cohere": None}):
            from synapsekit.llm.cohere import CohereLLM

            llm = CohereLLM(make_config("cohere", "command-r"))
            llm._client = None
            with pytest.raises(ImportError, match="cohere"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        delta_content = MagicMock()
        delta_content.text = "Hello"
        delta_msg = MagicMock()
        delta_msg.content = delta_content
        event = MagicMock()
        event.delta = MagicMock()
        event.delta.message = delta_msg

        async def mock_chat_stream(**kw):
            yield event

        mock_client = MagicMock()
        mock_client.chat_stream = mock_chat_stream
        mock_cohere = MagicMock()
        mock_cohere.AsyncClientV2.return_value = mock_client

        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            from synapsekit.llm.cohere import CohereLLM

            llm = CohereLLM(make_config("cohere"))
            tokens = []
            async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}]):
                tokens.append(t)
            assert tokens == ["Hello"]


# ------------------------------------------------------------------ #
# MistralLLM
# ------------------------------------------------------------------ #


class TestMistralLLM:
    def test_import_error_without_mistralai(self):
        with patch.dict("sys.modules", {"mistralai": None}):
            from synapsekit.llm.mistral import MistralLLM

            llm = MistralLLM(make_config("mistral", "mistral-small"))
            llm._client = None
            with pytest.raises(ImportError, match="mistralai"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        choice = MagicMock()
        choice.delta.content = "hi there"
        chunk_data = MagicMock()
        chunk_data.choices = [choice]
        chunk = MagicMock()
        chunk.data = chunk_data

        async def mock_stream_async(**kw):
            yield chunk

        mock_chat = MagicMock()
        mock_chat.stream_async = mock_stream_async
        mock_client = MagicMock()
        mock_client.chat = mock_chat
        mock_mistralai = MagicMock()
        mock_mistralai.Mistral.return_value = mock_client

        with patch.dict("sys.modules", {"mistralai": mock_mistralai}):
            from synapsekit.llm.mistral import MistralLLM

            llm = MistralLLM(make_config("mistral", "mistral-small"))
            tokens = []
            async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}]):
                tokens.append(t)
            assert tokens == ["hi there"]


# ------------------------------------------------------------------ #
# GeminiLLM
# ------------------------------------------------------------------ #


class TestGeminiLLM:
    def test_import_error_without_google_generativeai(self):
        with patch.dict("sys.modules", {"google.generativeai": None, "google": None}):
            from synapsekit.llm.gemini import GeminiLLM

            llm = GeminiLLM(make_config("gemini", "gemini-pro"))
            llm._model = None
            with pytest.raises((ImportError, AttributeError)):
                llm._get_model()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        chunk1 = MagicMock()
        chunk1.text = "Gemini"
        chunk2 = MagicMock()
        chunk2.text = " rocks"

        async def async_iter():
            for c in [chunk1, chunk2]:
                yield c

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=async_iter())

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.generativeai": mock_genai,
            },
        ):
            from synapsekit.llm.gemini import GeminiLLM

            llm = GeminiLLM(make_config("gemini", "gemini-pro"))
            llm._model = mock_model
            tokens = []
            async for t in llm.stream("hello"):
                tokens.append(t)
            assert tokens == ["Gemini", " rocks"]


# ------------------------------------------------------------------ #
# BedrockLLM
# ------------------------------------------------------------------ #


class TestBedrockLLM:
    def test_import_error_without_boto3(self):
        with patch.dict("sys.modules", {"boto3": None}):
            from synapsekit.llm.bedrock import BedrockLLM

            llm = BedrockLLM(make_config("bedrock", "anthropic.claude-v2"))
            llm._client = None
            with pytest.raises(ImportError, match="boto3"):
                llm._get_client()

    def test_build_body_claude(self):
        from synapsekit.llm.bedrock import BedrockLLM

        llm = BedrockLLM(make_config("bedrock", "anthropic.claude-v2"))
        body = llm._build_body("hello")
        assert "messages" in body
        assert body["anthropic_version"] == "bedrock-2023-05-31"

    def test_build_body_titan(self):
        from synapsekit.llm.bedrock import BedrockLLM

        llm = BedrockLLM(make_config("bedrock", "amazon.titan-text"))
        body = llm._build_body("hello")
        assert "inputText" in body

    def test_build_body_llama(self):
        from synapsekit.llm.bedrock import BedrockLLM

        llm = BedrockLLM(make_config("bedrock", "meta.llama2-13b"))
        body = llm._build_body("hello")
        assert "prompt" in body

    def test_extract_chunk_claude(self):
        from synapsekit.llm.bedrock import BedrockLLM

        llm = BedrockLLM(make_config("bedrock", "anthropic.claude-v2"))
        text = llm._extract_chunk({"delta": {"text": "hi"}}, "anthropic.claude-v2")
        assert text == "hi"

    def test_extract_chunk_titan(self):
        from synapsekit.llm.bedrock import BedrockLLM

        llm = BedrockLLM(make_config("bedrock", "amazon.titan-text"))
        text = llm._extract_chunk({"outputText": "yo"}, "amazon.titan-text")
        assert text == "yo"

    @pytest.mark.asyncio
    async def test_stream_with_messages(self):
        import json

        from synapsekit.llm.bedrock import BedrockLLM

        chunk_bytes = json.dumps({"delta": {"text": "AWS!"}}).encode()
        mock_event = {"chunk": {"bytes": chunk_bytes}}
        mock_response = {"body": [mock_event]}

        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.invoke_model_with_response_stream.return_value = mock_response
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            llm = BedrockLLM(make_config("bedrock", "anthropic.claude-v2"))
            tokens = []
            async for t in llm.stream_with_messages([{"role": "user", "content": "hi"}]):
                tokens.append(t)
            assert tokens == ["AWS!"]


# ------------------------------------------------------------------ #
# AI21LLM
# ------------------------------------------------------------------ #


class TestAI21LLM:
    def test_import_error_without_ai21(self):
        with patch.dict("sys.modules", {"ai21": None}):
            from synapsekit.llm.ai21 import AI21LLM

            llm = AI21LLM(make_config("ai21", "jamba-1.5-mini"))
            llm._client = None
            with pytest.raises(ImportError, match="ai21"):
                llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        chunk1.usage = None
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        chunk2.usage = None

        async def mock_stream():
            yield chunk1
            yield chunk2

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_ai21 = MagicMock()
        mock_ai21.AsyncAI21Client.return_value = mock_client

        with patch.dict("sys.modules", {"ai21": mock_ai21}):
            from synapsekit.llm.ai21 import AI21LLM

            llm = AI21LLM(make_config("ai21", "jamba-1.5-mini"))
            tokens = []
            async for t in llm.stream("hi"):
                tokens.append(t)
            assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_with_messages_passes_kwargs(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="ok"))]
        chunk.usage = None

        async def mock_stream():
            yield chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_ai21 = MagicMock()
        mock_ai21.AsyncAI21Client.return_value = mock_client

        with patch.dict("sys.modules", {"ai21": mock_ai21}):
            from synapsekit.llm.ai21 import AI21LLM

            llm = AI21LLM(make_config("ai21", "jamba-1.5-mini"))
            tokens = []
            async for t in llm.stream_with_messages(
                [{"role": "user", "content": "hi"}],
                temperature=0.8,
                max_tokens=100,
            ):
                tokens.append(t)
            assert tokens == ["ok"]
            kwargs = mock_client.chat.completions.create.call_args[1]
            assert kwargs["temperature"] == 0.8
            assert kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_call_with_tools(self):
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"location": "NYC"}'
        msg = MagicMock()
        msg.tool_calls = [tc]
        msg.content = None
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        mock_ai21 = MagicMock()
        mock_ai21.AsyncAI21Client.return_value = mock_client

        with patch.dict("sys.modules", {"ai21": mock_ai21}):
            from synapsekit.llm.ai21 import AI21LLM

            llm = AI21LLM(make_config("ai21", "jamba-1.5-mini"))
            result = await llm.call_with_tools(
                [{"role": "user", "content": "weather in NYC"}],
                [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            )
            assert result["tool_calls"][0]["name"] == "get_weather"
            assert result["tool_calls"][0]["arguments"] == {"location": "NYC"}


# ------------------------------------------------------------------ #
# DatabricksLLM
# ------------------------------------------------------------------ #


class TestDatabricksLLM:
    def test_import_error_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            from synapsekit.llm.databricks import DatabricksLLM

            llm = DatabricksLLM(
                make_config("databricks", "dbrx-instruct"), workspace_url="https://db.example.com"
            )
            llm._client = None
            with pytest.raises(ImportError, match="openai"):
                llm._get_client()

    def test_missing_workspace_url_raises(self):
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from synapsekit.llm.databricks import DatabricksLLM

            llm = DatabricksLLM(make_config("databricks", "dbrx-instruct"))
            with patch.dict("os.environ", {"DATABRICKS_HOST": ""}, clear=False):
                with pytest.raises(ValueError, match="workspace"):
                    llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"))]
        chunk1.usage = None
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content=" world"))]
        chunk2.usage = None

        async def mock_stream():
            yield chunk1
            yield chunk2

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        from synapsekit.llm.databricks import DatabricksLLM

        llm = DatabricksLLM(
            make_config("databricks", "dbrx-instruct"), workspace_url="https://db.example.com"
        )
        llm._client = mock_client
        tokens = []
        async for t in llm.stream("hi"):
            tokens.append(t)
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_with_messages_passes_kwargs(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="ok"))]
        chunk.usage = None

        async def mock_stream():
            yield chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        from synapsekit.llm.databricks import DatabricksLLM

        llm = DatabricksLLM(
            make_config("databricks", "dbrx-instruct"), workspace_url="https://db.example.com"
        )
        llm._client = mock_client
        tokens = []
        async for t in llm.stream_with_messages(
            [{"role": "user", "content": "hi"}],
            temperature=0.8,
            max_tokens=100,
        ):
            tokens.append(t)
        assert tokens == ["ok"]
        kwargs = mock_client.chat.completions.create.call_args[1]
        assert kwargs["temperature"] == 0.8
        assert kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_call_with_tools(self):
        tc = MagicMock()
        tc.id = "call_456"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"location": "SF"}'
        msg = MagicMock()
        msg.tool_calls = [tc]
        msg.content = None
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        from synapsekit.llm.databricks import DatabricksLLM

        llm = DatabricksLLM(
            make_config("databricks", "dbrx-instruct"), workspace_url="https://db.example.com"
        )
        llm._client = mock_client
        result = await llm.call_with_tools(
            [{"role": "user", "content": "weather in SF"}],
            [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        )
        assert result["tool_calls"][0]["name"] == "get_weather"


# ------------------------------------------------------------------ #
# ErnieLLM
# ------------------------------------------------------------------ #


class TestErnieLLM:
    def test_import_error_without_erniebot(self):
        with patch.dict("sys.modules", {"erniebot": None}):
            from synapsekit.llm.ernie import ErnieLLM

            llm = ErnieLLM(make_config("ernie", "ernie-3.5"))
            llm._configured = False
            with pytest.raises(ImportError, match="erniebot"):
                llm._configure()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        chunk1 = MagicMock()
        chunk1.get_result.return_value = "Hello"
        chunk2 = MagicMock()
        chunk2.get_result.return_value = " world"

        async def mock_stream():
            yield chunk1
            yield chunk2

        mock_erniebot = MagicMock()
        mock_erniebot.ChatCompletion.acreate.return_value = mock_stream()

        with patch.dict("sys.modules", {"erniebot": mock_erniebot}):
            from synapsekit.llm.ernie import ErnieLLM

            llm = ErnieLLM(make_config("ernie", "ernie-3.5"))
            llm._configured = True
            tokens = []
            async for t in llm.stream("hi"):
                tokens.append(t)
            assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_function_call(self):
        resp = MagicMock()
        resp.function_call = {"name": "get_weather", "arguments": '{"city": "Beijing"}'}
        resp.get_result.return_value = None

        mock_erniebot = MagicMock()
        mock_erniebot.ChatCompletion.acreate = AsyncMock(return_value=resp)

        with patch.dict("sys.modules", {"erniebot": mock_erniebot}):
            from synapsekit.llm.ernie import ErnieLLM

            llm = ErnieLLM(make_config("ernie", "ernie-3.5"))
            llm._configured = True
            result = await llm.call_with_tools(
                [{"role": "user", "content": "weather in Beijing"}],
                [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            )
            assert result["tool_calls"][0]["name"] == "get_weather"
            assert result["tool_calls"][0]["arguments"] == {"city": "Beijing"}

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        resp = MagicMock()
        resp.function_call = None
        resp.get_result.return_value = "The weather is sunny."

        mock_erniebot = MagicMock()
        mock_erniebot.ChatCompletion.acreate = AsyncMock(return_value=resp)

        with patch.dict("sys.modules", {"erniebot": mock_erniebot}):
            from synapsekit.llm.ernie import ErnieLLM

            llm = ErnieLLM(make_config("ernie", "ernie-3.5"))
            llm._configured = True
            result = await llm.call_with_tools(
                [{"role": "user", "content": "weather in Beijing"}],
                [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            )
            assert result["content"] == "The weather is sunny."
            assert result["tool_calls"] is None


# ------------------------------------------------------------------ #
# MinimaxLLM
# ------------------------------------------------------------------ #


class TestMinimaxLLM:
    def test_import_error_without_httpx(self):
        with patch.dict("sys.modules", {"httpx": None}):
            from synapsekit.llm.minimax import MinimaxLLM

            llm = MinimaxLLM(make_config("minimax", "abab5.5-chat"), group_id="group")
            llm._client = None
            with pytest.raises(ImportError, match="httpx"):
                llm._get_client()

    def test_missing_group_id_raises(self):
        mock_httpx = MagicMock()
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            from synapsekit.llm.minimax import MinimaxLLM

            llm = MinimaxLLM(make_config("minimax", "abab5.5-chat"))
            llm._client = None
            with patch("os.environ.get", return_value=None):
                with pytest.raises(ValueError, match="group_id"):
                    llm._get_client()

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        async def mock_aiter_lines():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " world"}}]}'
            yield "data: [DONE]"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream_cm

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            from synapsekit.llm.minimax import MinimaxLLM

            llm = MinimaxLLM(make_config("minimax", "abab5.5-chat"), group_id="group")
            tokens = []
            async for t in llm.stream("Hello"):
                tokens.append(t)
            assert tokens == ["Hello", " world"]
