"""VideoLoader — extract audio from video and transcribe."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from .audio import AudioLoader
from .base import Document

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VideoLoader:
    """Load video files by extracting audio and transcribing.

    Requires ``ffmpeg`` to be installed on the system.
    Delegates transcription to ``AudioLoader``.

    Example::

        loader = VideoLoader("lecture.mp4", api_key="sk-...")
        docs = loader.load()
    """

    def __init__(
        self,
        path: str,
        api_key: str | None = None,
        backend: str = "whisper_api",
        language: str | None = None,
        keep_audio: bool = False,
    ) -> None:
        self._path = Path(path)
        self._api_key = api_key
        self._backend = backend
        self._language = language
        self._keep_audio = keep_audio

        if self._path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: {self._path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def load(self) -> list[Document]:
        """Synchronously extract audio, transcribe, and return Documents."""
        audio_path = self._extract_audio_sync()
        try:
            loader = AudioLoader(
                path=str(audio_path),
                api_key=self._api_key,
                backend=self._backend,
                language=self._language,
            )
            docs = loader.load()
            # Update metadata
            for doc in docs:
                doc.metadata["source"] = str(self._path)
                doc.metadata["loader"] = "VideoLoader"
            return docs
        finally:
            if not self._keep_audio and audio_path.exists():
                audio_path.unlink()

    async def aload(self) -> list[Document]:
        """Async: extract audio via ffmpeg subprocess, then transcribe."""
        audio_path = await self._extract_audio_async()
        try:
            loader = AudioLoader(
                path=str(audio_path),
                api_key=self._api_key,
                backend=self._backend,
                language=self._language,
            )
            docs = await loader.aload()
            for doc in docs:
                doc.metadata["source"] = str(self._path)
                doc.metadata["loader"] = "VideoLoader"
            return docs
        finally:
            if not self._keep_audio and audio_path.exists():
                audio_path.unlink()

    def _extract_audio_sync(self) -> Path:
        """Extract audio using ffmpeg (sync subprocess)."""
        import subprocess

        audio_path = self._make_audio_path()
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(self._path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
        return audio_path

    async def _extract_audio_async(self) -> Path:
        """Extract audio using ffmpeg (async subprocess)."""
        audio_path = self._make_audio_path()
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            str(self._path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(audio_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
        return audio_path

    def _make_audio_path(self) -> Path:
        """Create a temp path for the extracted audio file."""
        suffix = ".wav"
        if self._keep_audio:
            return self._path.with_suffix(suffix)
        return Path(tempfile.mktemp(suffix=suffix, prefix="synapsekit_audio_"))
