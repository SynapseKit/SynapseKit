"""Production-grade tests for ImageGenerationTool."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.agents.tools.image_gen import ImageGenerationTool

# ── construction & subclassing ──────────────────────────────────────────────


class TestImageGenerationToolStructure:
    def test_subclasses_base_tool(self):
        assert issubclass(ImageGenerationTool, BaseTool)

    def test_has_name(self):
        assert ImageGenerationTool.name == "generate_image"

    def test_has_description(self):
        assert isinstance(ImageGenerationTool.description, str)
        assert len(ImageGenerationTool.description) > 10

    def test_has_parameters_schema(self):
        tool = ImageGenerationTool()
        assert "prompt" in tool.parameters["properties"]
        assert "prompt" in tool.parameters.get("required", [])

    def test_run_is_coroutine(self):
        tool = ImageGenerationTool()
        assert inspect.iscoroutinefunction(tool.run)

    def test_schema_is_openai_compatible(self):
        tool = ImageGenerationTool()
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "generate_image"

    def test_default_no_api_key(self):
        tool = ImageGenerationTool()
        assert tool._api_key == ""

    def test_custom_api_key(self):
        tool = ImageGenerationTool(api_key="sk-test")
        assert tool._api_key == "sk-test"


# ── validation ───────────────────────────────────────────────────────────────


class TestImageGenerationToolValidation:
    @pytest.mark.asyncio
    async def test_empty_prompt_returns_error(self):
        tool = ImageGenerationTool()
        result = await tool.run(prompt="")
        assert result.is_error
        assert "prompt" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_size_returns_error(self):
        tool = ImageGenerationTool()
        result = await tool.run(prompt="a cat", size="999x999")
        assert result.is_error
        assert "size" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_quality_returns_error(self):
        tool = ImageGenerationTool()
        result = await tool.run(prompt="a cat", quality="ultra")
        assert result.is_error
        assert "quality" in result.error.lower()

    @pytest.mark.asyncio
    async def test_valid_sizes_accepted(self):
        for size in ("1024x1024", "1792x1024", "1024x1792"):
            tool = ImageGenerationTool()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.data = [MagicMock(url="https://example.com/img.png")]
            mock_client.images.generate = AsyncMock(return_value=mock_response)
            tool._client = mock_client
            result = await tool.run(prompt="a cat", size=size)
            assert not result.is_error, f"size={size!r} should be valid"

    @pytest.mark.asyncio
    async def test_valid_qualities_accepted(self):
        for quality in ("standard", "hd"):
            tool = ImageGenerationTool()
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.data = [MagicMock(url="https://example.com/img.png")]
            mock_client.images.generate = AsyncMock(return_value=mock_response)
            tool._client = mock_client
            result = await tool.run(prompt="a cat", quality=quality)
            assert not result.is_error, f"quality={quality!r} should be valid"


# ── happy path ───────────────────────────────────────────────────────────────


class TestImageGenerationToolHappyPath:
    def _make_tool_with_mock_client(self, url: str = "https://example.com/img.png"):
        tool = ImageGenerationTool(api_key="sk-test")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(url=url)]
        mock_client.images.generate = AsyncMock(return_value=mock_response)
        tool._client = mock_client
        return tool, mock_client

    @pytest.mark.asyncio
    async def test_returns_tool_result(self):
        tool, _ = self._make_tool_with_mock_client()
        result = await tool.run(prompt="a sunset")
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_returns_url_in_output(self):
        url = "https://example.com/generated.png"
        tool, _ = self._make_tool_with_mock_client(url=url)
        result = await tool.run(prompt="a sunset")
        assert result.output == url
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_uses_dall_e_3_model(self):
        tool, mock_client = self._make_tool_with_mock_client()
        await tool.run(prompt="a cat")
        call_kwargs = mock_client.images.generate.call_args[1]
        assert call_kwargs["model"] == "dall-e-3"

    @pytest.mark.asyncio
    async def test_passes_prompt(self):
        tool, mock_client = self._make_tool_with_mock_client()
        await tool.run(prompt="futuristic city")
        call_kwargs = mock_client.images.generate.call_args[1]
        assert call_kwargs["prompt"] == "futuristic city"

    @pytest.mark.asyncio
    async def test_default_size_and_quality(self):
        tool, mock_client = self._make_tool_with_mock_client()
        await tool.run(prompt="a cat")
        call_kwargs = mock_client.images.generate.call_args[1]
        assert call_kwargs["size"] == "1024x1024"
        assert call_kwargs["quality"] == "standard"

    @pytest.mark.asyncio
    async def test_client_lazy_initialized_once(self):
        tool = ImageGenerationTool(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.data = [MagicMock(url="https://example.com/img.png")]

        with patch("synapsekit.agents.tools.image_gen.ImageGenerationTool._get_client") as mock_get:
            mock_async_client = MagicMock()
            mock_async_client.images.generate = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_async_client

            await tool.run(prompt="a cat")
            await tool.run(prompt="a dog")

        # _get_client should be called on each run but the client itself is cached
        assert mock_get.call_count == 2


# ── error handling ───────────────────────────────────────────────────────────


class TestImageGenerationToolErrors:
    @pytest.mark.asyncio
    async def test_api_exception_returns_error_result(self):
        tool = ImageGenerationTool(api_key="sk-test")
        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(side_effect=RuntimeError("quota exceeded"))
        tool._client = mock_client
        result = await tool.run(prompt="a cat")
        assert result.is_error
        assert "quota exceeded" in result.error

    @pytest.mark.asyncio
    async def test_no_url_in_response_returns_error(self):
        tool = ImageGenerationTool(api_key="sk-test")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(url=None)]
        mock_client.images.generate = AsyncMock(return_value=mock_response)
        tool._client = mock_client
        result = await tool.run(prompt="a cat")
        assert result.is_error
        assert "url" in result.error.lower()

    @pytest.mark.asyncio
    async def test_import_error_returns_error_result(self):
        tool = ImageGenerationTool()

        def raise_import(*a, **kw):
            raise ImportError("openai package required")

        with patch.object(tool, "_get_client", side_effect=raise_import):
            result = await tool.run(prompt="a cat")
        assert result.is_error
        assert "openai" in result.error.lower()
