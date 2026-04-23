"""AudioLoader — transcribe audio files via Whisper API or local Whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..text_splitters.sentence import SentenceTextSplitter
from .base import Document

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


class AudioLoader:
    """Load audio files by transcribing them into timestamped Documents.

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
        chunk_size: int = 6,
        chunk_overlap: int = 1,
    ) -> None:
        self._path = Path(path)
        self._api_key = api_key
        self._backend = backend
        self._language = language
        self._model = model
        self._splitter = SentenceTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if self._path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {self._path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def load(self) -> list[Document]:
        """Synchronously transcribe and return timestamped Documents."""
        if self._backend == "whisper_api":
            result = self._transcribe_api()
        elif self._backend == "whisper_local":
            result = self._transcribe_local()
        else:
            raise ValueError(f"Unknown backend: {self._backend!r}")

        return self._to_documents(result)

    async def aload(self) -> list[Document]:
        """Async transcription (wraps sync transcription in a worker thread)."""
        import asyncio

        return await asyncio.to_thread(self.load)

    def _transcribe_api(self) -> dict[str, Any]:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for whisper_api backend. "
                "Install it with: pip install 'synapsekit[audio]'"
            ) from None

        client = openai.OpenAI(api_key=self._api_key)
        with open(self._path, "rb") as audio_file:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "file": audio_file,
                "response_format": "verbose_json",
            }
            if self._language:
                kwargs["language"] = self._language
            transcript = client.audio.transcriptions.create(**kwargs)

        text = str(getattr(transcript, "text", "") or "")
        raw_segments = getattr(transcript, "segments", None) or []
        segments = self._normalise_segments(raw_segments)
        return {"text": text, "segments": segments}

    def _transcribe_local(self) -> dict[str, Any]:
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is required for whisper_local backend. "
                "Install it with: pip install openai-whisper"
            ) from None

        model_name = self._model if self._model != "whisper-1" else "base"
        model = whisper.load_model(model_name)
        kwargs: dict[str, Any] = {}
        if self._language:
            kwargs["language"] = self._language

        result = model.transcribe(str(self._path), **kwargs)
        text = str(result.get("text", ""))
        segments = self._normalise_segments(result.get("segments", []))
        return {"text": text, "segments": segments}

    def _normalise_segments(self, raw_segments: Any) -> list[dict[str, Any]]:
        normalised: list[dict[str, Any]] = []
        for seg in raw_segments or []:
            text = str(self._segment_value(seg, "text", "") or "").strip()
            if not text:
                continue
            start = self._to_float(self._segment_value(seg, "start", None))
            end = self._to_float(self._segment_value(seg, "end", None))
            normalised.append({"text": text, "start": start, "end": end})
        return normalised

    @staticmethod
    def _segment_value(seg: Any, key: str, default: Any) -> Any:
        if isinstance(seg, dict):
            return seg.get(key, default)
        return getattr(seg, key, default)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_documents(self, transcription: dict[str, Any]) -> list[Document]:
        base_metadata = {
            "source": str(self._path),
            "file": str(self._path),
            "source_type": "audio",
            "loader": "AudioLoader",
            "backend": self._backend,
        }

        docs: list[Document] = []
        for seg in transcription.get("segments", []):
            segment_text = str(seg.get("text", "")).strip()
            if not segment_text:
                continue
            chunks = self._splitter.split(segment_text) or [segment_text]
            for chunk in chunks:
                docs.append(
                    Document(
                        text=chunk,
                        metadata={
                            **base_metadata,
                            "start_time": seg.get("start"),
                            "end_time": seg.get("end"),
                        },
                    )
                )

        if docs:
            return docs

        text = str(transcription.get("text", "") or "").strip()
        if not text:
            return []

        chunks = self._splitter.split(text) or [text]
        return [Document(text=chunk, metadata=dict(base_metadata)) for chunk in chunks]
