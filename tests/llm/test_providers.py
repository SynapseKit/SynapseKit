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
        mock_chunk1 = MagicMock()
        mock_chunk1.get_result.return_value = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.get_result.return_value = " world"

        async def mock_acreate(**kw):
            for chunk in [mock_chunk1, mock_chunk2]:
                yield chunk

        mock_erniebot = MagicMock()
        mock_erniebot.ChatCompletion.acreate = mock_acreate

        with patch.dict("sys.modules", {"erniebot": mock_erniebot}):
            from synapsekit.llm.ernie import ErnieLLM

            llm = ErnieLLM(make_config("ernie", "ernie-3.5"))
            tokens = []
            async for t in llm.stream("Hello"):
                tokens.append(t)
            assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_function_call(self):
        mock_response = MagicMock()
        mock_response.function_call = {
            "name": "get_weather",
            "arguments": '{"city": "Beijing"}',
        }
        mock_response.get_result.return_value = None

        async def mock_acreate(**kw):
            return mock_response

        mock_erniebot = MagicMock()
        mock_erniebot.ChatCompletion.acreate = mock_acreate

        with patch.dict("sys.modules", {"erniebot": mock_erniebot}):
            from synapsekit.llm.ernie import ErnieLLM

            llm = ErnieLLM(make_config("ernie", "ernie-3.5"))
            result = await llm._call_with_tools_impl(
                [{"role": "user", "content": "What's the weather?"}],
                [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            )
            assert result["tool_calls"] is not None
            assert result["tool_calls"][0]["name"] == "get_weather"
            assert result["tool_calls"][0]["arguments"] == {"city": "Beijing"}

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        mock_response = MagicMock()
        mock_response.function_call = None
        mock_response.get_result.return_value = "The weather is sunny."

        async def mock_acreate(**kw):
            return mock_response

        mock_erniebot = MagicMock()
        mock_erniebot.ChatCompletion.acreate = mock_acreate

        with patch.dict("sys.modules", {"erniebot": mock_erniebot}):
            from synapsekit.llm.ernie import ErnieLLM

            llm = ErnieLLM(make_config("ernie", "ernie-3.5"))
            result = await llm._call_with_tools_impl(
                [{"role": "user", "content": "What's the weather?"}],
                [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            )
            assert result["content"] == "The weather is sunny."
            assert result["tool_calls"] is None
