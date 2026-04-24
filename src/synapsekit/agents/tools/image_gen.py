"""Image Generation Tool: generate images with DALL-E via the OpenAI API."""

from __future__ import annotations

from typing import Any

from ..base import BaseTool, ToolResult


class ImageGenerationTool(BaseTool):
    """Generate an image from a text prompt using OpenAI's DALL-E 3.

    Usage::

        tool = ImageGenerationTool(api_key="sk-...")
        result = await tool.run(prompt="a red panda on a skateboard")
        # result.output contains the image URL
    """

    name = "generate_image"
    description = (
        "Generate an image from a text prompt using DALL-E 3. "
        "Returns a URL pointing to the generated image."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the image to generate.",
            },
            "size": {
                "type": "string",
                "description": "Image dimensions. One of: 1024x1024, 1792x1024, 1024x1792.",
                "default": "1024x1024",
            },
            "quality": {
                "type": "string",
                "description": "Image quality. One of: standard, hd.",
                "default": "standard",
            },
        },
        "required": ["prompt"],
    }

    _VALID_SIZES = frozenset({"1024x1024", "1792x1024", "1024x1792"})
    _VALID_QUALITIES = frozenset({"standard", "hd"})

    def __init__(self, api_key: str = "", base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai package required: pip install openai") from None
            self._client = AsyncOpenAI(
                api_key=self._api_key or None,
                base_url=self._base_url,
            )
        return self._client

    async def run(
        self,
        prompt: str = "",
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs: Any,
    ) -> ToolResult:
        if not prompt:
            return ToolResult(output="", error="prompt must not be empty.")
        if size not in self._VALID_SIZES:
            return ToolResult(
                output="",
                error=f"Invalid size {size!r}. Choose from: {', '.join(sorted(self._VALID_SIZES))}",
            )
        if quality not in self._VALID_QUALITIES:
            return ToolResult(
                output="",
                error=f"Invalid quality {quality!r}. Choose from: {', '.join(sorted(self._VALID_QUALITIES))}",
            )

        try:
            client = self._get_client()
            response = await client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,  # type: ignore[arg-type]
                quality=quality,  # type: ignore[arg-type]
                n=1,
            )
            url = response.data[0].url
            if not url:
                return ToolResult(output="", error="OpenAI returned no image URL.")
            return ToolResult(output=url)
        except ImportError as exc:
            return ToolResult(output="", error=str(exc))
        except Exception as exc:
            return ToolResult(output="", error=f"Image generation failed: {exc}")
