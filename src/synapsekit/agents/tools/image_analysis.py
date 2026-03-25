"""Image Analysis Tool: describe or analyze an image with a multimodal LLM."""

from __future__ import annotations

from typing import Any

from ...llm.multimodal import ImageContent, MultimodalMessage
from ..base import BaseTool, ToolResult


class ImageAnalysisTool(BaseTool):
    """Analyze an image using a multimodal LLM.

    Usage::

        tool = ImageAnalysisTool(llm)
        result = await tool.run(path="/path/to/image.png")
    """

    name = "image_analysis"
    description = (
        "Analyze an image using a multimodal LLM. "
        "Input: image path or image URL, and an optional prompt. "
        "Returns: a text description or analysis of the image."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local image file path",
            },
            "image_url": {
                "type": "string",
                "description": "Public image URL",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt describing the analysis to perform",
                "default": "Describe this image in detail.",
            },
            "media_type": {
                "type": "string",
                "description": "Media type for image_url (default: image/png)",
                "default": "image/png",
            },
        },
    }

    def __init__(self, llm: Any, default_prompt: str = "Describe this image in detail.") -> None:
        self._llm = llm
        self._default_prompt = default_prompt

    async def run(
        self,
        path: str | None = None,
        image_url: str | None = None,
        prompt: str | None = None,
        media_type: str = "image/png",
        **kwargs: Any,
    ) -> ToolResult:
        img_path = path or kwargs.get("input")
        img_url = image_url or kwargs.get("url")
        if not img_path and not img_url:
            return ToolResult(output="", error="Provide an image path or image URL.")

        if self._llm is None:
            return ToolResult(output="", error="No LLM provided for image analysis.")

        analysis_prompt = prompt or self._default_prompt

        try:
            if img_path:
                image = ImageContent.from_file(img_path)
            elif img_url:
                image = ImageContent.from_url(img_url, media_type=media_type)
            else:
                return ToolResult(output="", error="Provide an image path or image URL.")

            message = MultimodalMessage(text=analysis_prompt, images=[image])
            provider = getattr(getattr(self._llm, "config", None), "provider", "openai")
            if provider == "anthropic":
                messages = message.to_anthropic_messages()
            else:
                messages = message.to_openai_messages()

            result = await self._llm.generate_with_messages(messages)
            return ToolResult(output=str(result).strip())
        except Exception as e:
            return ToolResult(output="", error=f"Image analysis failed: {e}")
