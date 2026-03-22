"""AudioLoader — transcribe audio files via Whisper API or local Whisper."""

from __future__ import annotations

from pathlib import Path

from .base import Document

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


class AudioLoader:
    """Load audio files by transcribing them into Documents.

    Backends:

    - ``"whisper_api"`` (default) — uses the OpenAI Whisper API (requires ``openai``)
    - ``"whisper_local"`` — uses local ``openai-whisper`` package

    Example::

        loader = AudioLoader("interview.mp3", api_key="sk-...")
        docs = loader.load()
    """

    def __init__(
        self,
        path: str,
        api_key: str | None = None,
        backend: str = "whisper_api",
        language: str | None = None,
        model: str = "whisper-1",
    ) -> None:
        self._path = Path(path)
        self._api_key = api_key
        self._backend = backend
        self._language = language
        self._model = model

        if self._path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {self._path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def load(self) -> list[Document]:
        """Synchronously transcribe and return as Documents."""
        if self._backend == "whisper_api":
            text = self._transcribe_api()
        elif self._backend == "whisper_local":
            text = self._transcribe_local()
        else:
            raise ValueError(f"Unknown backend: {self._backend!r}")

        return [
            Document(
                text=text,
                metadata={
                    "source": str(self._path),
                    "loader": "AudioLoader",
                    "backend": self._backend,
                },
            )
        ]

    async def aload(self) -> list[Document]:
        """Async transcription (wraps sync for API backend)."""
        import asyncio

        return await asyncio.to_thread(self.load)

    def _transcribe_api(self) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for whisper_api backend. "
                "Install it with: pip install 'synapsekit[audio]'"
            ) from None

        client = openai.OpenAI(api_key=self._api_key)
        with open(self._path, "rb") as audio_file:
            kwargs: dict = {"model": self._model, "file": audio_file}
            if self._language:
                kwargs["language"] = self._language
            transcript = client.audio.transcriptions.create(**kwargs)
        return str(transcript.text)

    def _transcribe_local(self) -> str:
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is required for whisper_local backend. "
                "Install it with: pip install openai-whisper"
            ) from None

        model = whisper.load_model(self._model if self._model != "whisper-1" else "base")
        kwargs: dict = {}
        if self._language:
            kwargs["language"] = self._language
        result = model.transcribe(str(self._path), **kwargs)
        return str(result["text"])
