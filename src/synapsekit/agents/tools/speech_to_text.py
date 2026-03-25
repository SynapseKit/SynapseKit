"""Speech-to-Text Tool: transcribe audio files."""

from __future__ import annotations

from typing import Any

from ...loaders.audio import AudioLoader
from ..base import BaseTool, ToolResult


class SpeechToTextTool(BaseTool):
    """Transcribe an audio file into text.

    Usage::

        tool = SpeechToTextTool(api_key="sk-...")
        result = await tool.run(path="/path/to/audio.mp3")
    """

    name = "speech_to_text"
    description = (
        "Transcribe an audio file into text. "
        "Input: audio file path and optional backend/model/language. "
        "Returns: the transcription text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Audio file path",
            },
            "backend": {
                "type": "string",
                "description": "Backend: 'whisper_api' or 'whisper_local' (default: whisper_api)",
                "default": "whisper_api",
            },
            "language": {
                "type": "string",
                "description": "Optional language hint (e.g., 'en')",
            },
            "model": {
                "type": "string",
                "description": "Model name (default: whisper-1 for API, base for local)",
                "default": "whisper-1",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        api_key: str | None = None,
        backend: str = "whisper_api",
        model: str = "whisper-1",
        language: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._backend = backend
        self._model = model
        self._language = language

    async def run(self, path: str = "", **kwargs: Any) -> ToolResult:
        audio_path = path or kwargs.get("input", "")
        if not audio_path:
            return ToolResult(output="", error="No audio file path provided.")

        backend = kwargs.get("backend", self._backend)
        language = kwargs.get("language", self._language)
        model = kwargs.get("model", self._model)
        api_key = kwargs.get("api_key", self._api_key)

        try:
            loader = AudioLoader(
                path=audio_path,
                api_key=api_key,
                backend=backend,
                language=language,
                model=model,
            )
            docs = await loader.aload()
            text = docs[0].text if docs else ""
            return ToolResult(output=text)
        except Exception as e:
            return ToolResult(output="", error=f"Speech-to-text failed: {e}")
