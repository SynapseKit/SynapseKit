"""Tests for VertexAILLM provider — mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.llm.base import LLMConfig


def make_config(provider="vertex", model="gemini-1.5-pro"):
    return LLMConfig(
        model=model,
        api_key="my-gcp-project",
        provider=provider,
        system_prompt="You are helpful.",
        temperature=0.2,
        max_tokens=100,
    )


async def _async_chunks(texts):
    """Helper to create an async iterable of mock chunks."""
    for t in texts:
        chunk = MagicMock()
        chunk.text = t
        yield chunk


class TestVertexAILLM:
    def test_lazy_init(self):
        """Verify _model is None initially."""
        with patch.dict(
            "sys.modules",
            {
                "vertexai": MagicMock(),
                "vertexai.generative_models": MagicMock(),
            },
        ):
            from synapsekit.llm.vertex_ai import VertexAILLM

            llm = VertexAILLM(make_config())
            assert llm._model is None

    def test_get_model_imports_vertexai(self):
        """Patch vertexai imports, verify init called with project."""
        mock_vertexai = MagicMock()
        mock_gen_models = MagicMock()
        mock_model_instance = MagicMock()
        mock_gen_models.GenerativeModel.return_value = mock_model_instance

        with patch.dict(
            "sys.modules",
            {
                "vertexai": mock_vertexai,
                "vertexai.generative_models": mock_gen_models,
            },
        ):
            from synapsekit.llm.vertex_ai import VertexAILLM

            llm = VertexAILLM(make_config())
            model = llm._get_model()

            mock_vertexai.init.assert_called_once_with(project="my-gcp-project")
            mock_gen_models.GenerativeModel.assert_called_once_with(
                model_name="gemini-1.5-pro",
                system_instruction="You are helpful.",
            )
            assert model is mock_model_instance

    @pytest.mark.asyncio
    async def test_stream(self):
        """Mock model.generate_content_async to return async iterable of chunks."""
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(
            return_value=_async_chunks(["Hello", " World"])
        )

        mock_vertexai = MagicMock()
        mock_gen_models = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "vertexai": mock_vertexai,
                "vertexai.generative_models": mock_gen_models,
            },
        ):
            from synapsekit.llm.vertex_ai import VertexAILLM

            llm = VertexAILLM(make_config())
            llm._model = mock_model
            tokens = []
            async for t in llm.stream("hello"):
                tokens.append(t)
            assert tokens == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_stream_with_messages(self):
        """Verify _messages_to_prompt called then stream."""
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(
            return_value=_async_chunks(["ok"])
        )

        mock_vertexai = MagicMock()
        mock_gen_models = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "vertexai": mock_vertexai,
                "vertexai.generative_models": mock_gen_models,
            },
        ):
            from synapsekit.llm.vertex_ai import VertexAILLM

            llm = VertexAILLM(make_config())
            llm._model = mock_model
            tokens = []
            async for t in llm.stream_with_messages(
                [{"role": "user", "content": "hi"}]
            ):
                tokens.append(t)
            assert tokens == ["ok"]

    @pytest.mark.asyncio
    async def test_call_with_tools_text_response(self):
        """Mock response with text parts, verify content returned."""
        text_part = MagicMock()
        text_part.function_call.name = ""
        text_part.text = "Just text"

        candidate = MagicMock()
        candidate.content.parts = [text_part]
        mock_response = MagicMock()
        mock_response.candidates = [candidate]

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        mock_vertexai = MagicMock()
        mock_gen_models = MagicMock()
        mock_aiplatform_v1 = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "vertexai": mock_vertexai,
                "vertexai.generative_models": mock_gen_models,
                "google": MagicMock(),
                "google.cloud": MagicMock(),
                "google.cloud.aiplatform_v1": mock_aiplatform_v1,
                "google.cloud.aiplatform_v1.types": MagicMock(),
                "google.cloud.aiplatform_v1.types.content": MagicMock(),
            },
        ):
            from synapsekit.llm.vertex_ai import VertexAILLM

            llm = VertexAILLM(make_config())
            llm._model = mock_model

            result = await llm._call_with_tools_impl(
                messages=[{"role": "user", "content": "hello"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "City name",
                                    }
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ],
            )

            assert result["content"] == "Just text"
            assert result["tool_calls"] is None

    @pytest.mark.asyncio
    async def test_call_with_tools_function_call(self):
        """Mock response with function_call parts, verify tool_calls returned."""
        fc_part = MagicMock()
        fc_part.function_call.name = "get_weather"
        fc_part.function_call.args = {"city": "London"}
        # Make sure hasattr(part, "text") check doesn't match for fc
        del fc_part.text

        candidate = MagicMock()
        candidate.content.parts = [fc_part]
        mock_response = MagicMock()
        mock_response.candidates = [candidate]

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        mock_vertexai = MagicMock()
        mock_gen_models = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "vertexai": mock_vertexai,
                "vertexai.generative_models": mock_gen_models,
                "google": MagicMock(),
                "google.cloud": MagicMock(),
                "google.cloud.aiplatform_v1": MagicMock(),
                "google.cloud.aiplatform_v1.types": MagicMock(),
                "google.cloud.aiplatform_v1.types.content": MagicMock(),
            },
        ):
            from synapsekit.llm.vertex_ai import VertexAILLM

            llm = VertexAILLM(make_config())
            llm._model = mock_model

            result = await llm._call_with_tools_impl(
                messages=[{"role": "user", "content": "weather in London"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "City name",
                                    }
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ],
            )

            assert result["content"] is None
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["name"] == "get_weather"
            assert result["tool_calls"][0]["arguments"] == {"city": "London"}
            assert result["tool_calls"][0]["id"].startswith("call_")

    def test_import_error(self):
        """Verify ImportError message when vertexai not installed."""
        with patch.dict(
            "sys.modules",
            {
                "vertexai": None,
                "vertexai.generative_models": None,
            },
        ):
            from synapsekit.llm.vertex_ai import VertexAILLM

            llm = VertexAILLM(make_config())
            llm._model = None
            with pytest.raises(ImportError, match="google-cloud-aiplatform"):
                llm._get_model()

    def test_convert_params(self):
        """Test the static method directly."""
        from synapsekit.llm.vertex_ai import VertexAILLM

        # Empty params
        assert VertexAILLM._convert_params({}) == {}

        # Full params
        params = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "description": "Temperature units"},
            },
            "required": ["city"],
        }
        result = VertexAILLM._convert_params(params)
        assert result["type"] == "OBJECT"
        assert result["properties"]["city"]["type"] == "STRING"
        assert result["properties"]["city"]["description"] == "City name"
        assert result["properties"]["units"]["type"] == "STRING"
        assert result["required"] == ["city"]
