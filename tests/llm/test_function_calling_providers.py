"""Tests for call_with_tools() on Gemini and Mistral providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
]

MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is the weather in Paris?"},
]


def _config(provider: str) -> LLMConfig:
    return LLMConfig(model="test-model", api_key="test-key", provider=provider)


# ------------------------------------------------------------------ #
# MistralLLM.call_with_tools()
# ------------------------------------------------------------------ #


class TestMistralCallWithTools:
    @pytest.fixture
    def llm(self):
        from synapsekit.llm.mistral import MistralLLM

        return MistralLLM(_config("mistral"))

    async def test_returns_tool_calls(self, llm):
        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "get_weather"
        mock_tc.function.arguments = '{"city": "Paris"}'

        mock_msg = MagicMock()
        mock_msg.tool_calls = [mock_tc]
        mock_msg.content = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_msg)]

        mock_client = MagicMock()
        mock_client.chat.complete_async = AsyncMock(return_value=mock_response)
        llm._client = mock_client

        result = await llm.call_with_tools(MESSAGES, TOOL_SCHEMA)
        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][0]["arguments"] == {"city": "Paris"}
        assert result["content"] is None

    async def test_returns_text_when_no_tools(self, llm):
        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content = "The weather is sunny."

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_msg)]

        mock_client = MagicMock()
        mock_client.chat.complete_async = AsyncMock(return_value=mock_response)
        llm._client = mock_client

        result = await llm.call_with_tools(MESSAGES, TOOL_SCHEMA)
        assert result["content"] == "The weather is sunny."
        assert result["tool_calls"] is None


# ------------------------------------------------------------------ #
# GeminiLLM.call_with_tools()
# ------------------------------------------------------------------ #


class TestGeminiCallWithTools:
    @pytest.fixture
    def llm(self):
        from synapsekit.llm.gemini import GeminiLLM

        return GeminiLLM(_config("gemini"))

    async def test_returns_tool_calls(self, llm):
        mock_fc = MagicMock()
        mock_fc.name = "get_weather"
        mock_fc.args = {"city": "Paris"}

        mock_part = MagicMock()
        mock_part.function_call = mock_fc
        mock_part.text = ""

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        llm._model = mock_model

        mock_genai = MagicMock()
        mock_genai.protos.FunctionDeclaration.return_value = MagicMock()
        mock_genai.protos.Tool.return_value = MagicMock()

        with patch.dict("sys.modules", {"google.generativeai": mock_genai, "google": MagicMock()}):
            result = await llm.call_with_tools(MESSAGES, TOOL_SCHEMA)

        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][0]["arguments"] == {"city": "Paris"}
        assert result["tool_calls"][0]["id"].startswith("call_")
        assert result["content"] is None

    async def test_returns_text_when_no_tools(self, llm):
        # Create a part object where hasattr(part, "function_call") is False
        mock_part_obj = MagicMock()
        del mock_part_obj.function_call  # Remove attribute so hasattr returns False
        mock_part_obj.text = "It is sunny in Paris."

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part_obj]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        llm._model = mock_model

        mock_genai = MagicMock()
        mock_genai.protos.FunctionDeclaration.return_value = MagicMock()
        mock_genai.protos.Tool.return_value = MagicMock()

        with patch.dict("sys.modules", {"google.generativeai": mock_genai, "google": MagicMock()}):
            result = await llm.call_with_tools(MESSAGES, TOOL_SCHEMA)

        assert result["content"] == "It is sunny in Paris."
        assert result["tool_calls"] is None


# ------------------------------------------------------------------ #
# Error message update
# ------------------------------------------------------------------ #


def test_function_calling_error_message_includes_new_providers():
    from synapsekit.agents.function_calling import FunctionCallingAgent
    from synapsekit.llm.base import BaseLLM

    class DummyLLM(BaseLLM):
        async def stream(self, prompt, **kw):
            yield "x"

    llm = DummyLLM(_config("dummy"))
    agent = FunctionCallingAgent(llm=llm, tools=[])
    with pytest.raises(RuntimeError, match="GeminiLLM / MistralLLM"):
        agent._check_support()
