"""Text-to-Speech Tool: synthesize audio from text."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..base import BaseTool, ToolResult


class TextToSpeechTool(BaseTool):
    """Synthesize speech audio from text using OpenAI TTS.

    Usage::

        tool = TextToSpeechTool(api_key="sk-...")
        result = await tool.run(text="Hello", output_path="/tmp/hello.mp3")
    """

    name = "text_to_speech"
    description = (
        "Convert text to speech audio. "
        "Input: text and output_path, with optional voice/model/format. "
        "Returns: a confirmation with the saved audio path."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to synthesize",
            },
            "output_path": {
                "type": "string",
                "description": "Where to save the audio file",
            },
            "voice": {
                "type": "string",
                "description": "Voice name (default: alloy)",
                "default": "alloy",
            },
            "model": {
                "type": "string",
                "description": "TTS model (default: tts-1)",
                "default": "tts-1",
            },
            "format": {
                "type": "string",
                "description": "Audio format (mp3, wav, flac, aac) (default: mp3)",
                "default": "mp3",
            },
        },
        "required": ["text", "output_path"],
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def run(
        self,
        text: str = "",
        output_path: str | None = None,
        voice: str = "alloy",
        model: str = "tts-1",
        format: str = "mp3",
        **kwargs: Any,
    ) -> ToolResult:
        input_text = text or kwargs.get("input", "")
        if not input_text:
            return ToolResult(output="", error="No text provided for TTS.")

        output_path = output_path or kwargs.get("path")
        if not output_path:
            return ToolResult(output="", error="No output_path provided.")

        api_key = kwargs.get("api_key") or self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ToolResult(output="", error="No OPENAI_API_KEY configured.")

        try:
            import openai
        except ImportError:
            return ToolResult(
                output="",
                error="openai package required for TTS. Install with: pip install synapsekit[openai]",
            )

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        def _synthesize() -> None:
            client = openai.OpenAI(api_key=api_key)
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=input_text,
                response_format=format,
            )
            response.stream_to_file(str(target))

        try:
            import asyncio

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _synthesize)
        except Exception as e:
            return ToolResult(output="", error=f"Text-to-speech failed: {e}")

        return ToolResult(output=f"Saved speech audio to {target}")
