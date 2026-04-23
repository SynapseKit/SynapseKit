"""VideoLoader — extract transcript + frame captions from video files."""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from .._compat import run_sync
from .audio import AudioLoader
from .base import Document
from .image import ImageLoader

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VideoLoader:
    """Load video files by extracting transcript chunks and frame-caption chunks.

    Requires ``ffmpeg`` to be installed on the system.
    Audio transcription is delegated to ``AudioLoader``.

    Example::

        loader = VideoLoader("lecture.mp4", api_key="sk-...", frame_interval=30)
        docs = await loader.aload()
    """

    def __init__(
        self,
        path: str,
        api_key: str | None = None,
        backend: str = "whisper_api",
        language: str | None = None,
        keep_audio: bool = False,
        llm: Any = None,
        frame_interval: float | int | None = 30,
        frame_prompt: str = "Describe this video frame in detail for retrieval.",
        keep_frames: bool = False,
    ) -> None:
        self._path = Path(path)
        self._api_key = api_key
        self._backend = backend
        self._language = language
        self._keep_audio = keep_audio
        self._llm = llm
        self._frame_interval = float(frame_interval) if frame_interval else None
        self._frame_prompt = frame_prompt
        self._keep_frames = keep_frames

        if self._path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format: {self._path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def load(self) -> list[Document]:
        """Synchronously extract audio/frame content and return Documents."""
        audio_path = self._extract_audio_sync()
        frame_paths = self._extract_frames_sync()
        try:
            transcript_docs = self._load_transcript_docs_sync(audio_path)
            frame_docs = self._frame_documents_sync(frame_paths)
            return transcript_docs + frame_docs
        finally:
            self._cleanup_audio_file(audio_path)
            self._cleanup_frame_files(frame_paths)

    async def aload(self) -> list[Document]:
        """Async: extract audio/frame content and return Documents."""
        audio_path = await self._extract_audio_async()
        frame_paths = await self._extract_frames_async()
        try:
            transcript_docs = await self._load_transcript_docs_async(audio_path)
            frame_docs = await self._frame_documents_async(frame_paths)
            return transcript_docs + frame_docs
        finally:
            self._cleanup_audio_file(audio_path)
            self._cleanup_frame_files(frame_paths)

    def _load_transcript_docs_sync(self, audio_path: Path) -> list[Document]:
        loader = AudioLoader(
            path=str(audio_path),
            api_key=self._api_key,
            backend=self._backend,
            language=self._language,
        )
        docs = loader.load()
        return [self._decorate_transcript_doc(doc) for doc in docs]

    async def _load_transcript_docs_async(self, audio_path: Path) -> list[Document]:
        loader = AudioLoader(
            path=str(audio_path),
            api_key=self._api_key,
            backend=self._backend,
            language=self._language,
        )
        docs = await loader.aload()
        return [self._decorate_transcript_doc(doc) for doc in docs]

    def _decorate_transcript_doc(self, doc: Document) -> Document:
        timestamp = doc.metadata.get("start_time")
        doc.metadata = {
            **doc.metadata,
            "source": str(self._path),
            "file": str(self._path),
            "source_type": "video",
            "loader": "VideoLoader",
            "chunk_type": "transcript",
            "timestamp": self._to_float(timestamp),
        }
        return doc

    def _frame_documents_sync(self, frame_paths: list[Path]) -> list[Document]:
        docs: list[Document] = []
        for idx, frame_path in enumerate(frame_paths, start=1):
            loader = ImageLoader(path=frame_path, llm=self._llm, prompt=self._frame_prompt)
            frame_docs = run_sync(loader.aload())
            docs.extend(self._decorate_frame_docs(frame_docs, frame_path, idx))
        return docs

    async def _frame_documents_async(self, frame_paths: list[Path]) -> list[Document]:
        docs: list[Document] = []
        for idx, frame_path in enumerate(frame_paths, start=1):
            loader = ImageLoader(path=frame_path, llm=self._llm, prompt=self._frame_prompt)
            frame_docs = await loader.aload()
            docs.extend(self._decorate_frame_docs(frame_docs, frame_path, idx))
        return docs

    def _decorate_frame_docs(
        self,
        docs: list[Document],
        frame_path: Path,
        frame_index: int,
    ) -> list[Document]:
        timestamp = self._frame_timestamp(frame_index)
        for doc in docs:
            doc.metadata = {
                **doc.metadata,
                "source": str(self._path),
                "file": str(self._path),
                "source_type": "video",
                "loader": "VideoLoader",
                "chunk_type": "frame_caption",
                "timestamp": timestamp,
                "frame_path": str(frame_path),
                "frame_index": frame_index,
            }
        return docs

    def _frame_timestamp(self, frame_index: int) -> float | None:
        if not self._frame_interval:
            return None
        return (frame_index - 1) * self._frame_interval

    def _extract_audio_sync(self) -> Path:
        """Extract mono 16kHz WAV audio using ffmpeg (sync subprocess)."""
        import subprocess

        audio_path = self._make_audio_path()
        cmd = [
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
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg is required for VideoLoader but was not found.") from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
            raise RuntimeError(f"ffmpeg failed to extract audio: {stderr}") from e
        return audio_path

    async def _extract_audio_async(self) -> Path:
        """Extract mono 16kHz WAV audio using ffmpeg (async subprocess)."""
        audio_path = self._make_audio_path()
        cmd = [
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
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg is required for VideoLoader but was not found.") from e

        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to extract audio: {stderr.decode(errors='ignore')}")
        return audio_path

    def _extract_frames_sync(self) -> list[Path]:
        """Extract frame images with ffmpeg at configured intervals."""
        if not self._frame_interval or self._frame_interval <= 0:
            return []

        import subprocess

        frames_dir = Path(tempfile.mkdtemp(prefix="synapsekit_frames_"))
        frame_pattern = frames_dir / "frame_%06d.jpg"
        cmd = [
            "ffmpeg",
            "-i",
            str(self._path),
            "-vf",
            f"fps=1/{self._frame_interval}",
            "-q:v",
            "2",
            "-y",
            str(frame_pattern),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError as e:
            self._remove_dir_if_exists(frames_dir)
            raise RuntimeError("ffmpeg is required for VideoLoader but was not found.") from e
        except subprocess.CalledProcessError as e:
            self._remove_dir_if_exists(frames_dir)
            stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
            raise RuntimeError(f"ffmpeg failed to extract frames: {stderr}") from e

        return sorted(frames_dir.glob("frame_*.jpg"))

    async def _extract_frames_async(self) -> list[Path]:
        """Extract frames asynchronously by offloading sync extraction."""
        return await asyncio.to_thread(self._extract_frames_sync)

    def _make_audio_path(self) -> Path:
        """Create a temp path for the extracted audio file."""
        suffix = ".wav"
        if self._keep_audio:
            return self._path.with_suffix(suffix)
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix="synapsekit_audio_",
            delete=False,
        ) as fd:
            return Path(fd.name)

    def _cleanup_audio_file(self, audio_path: Path) -> None:
        if self._keep_audio:
            return
        if audio_path.exists():
            audio_path.unlink()

    def _cleanup_frame_files(self, frame_paths: list[Path]) -> None:
        if self._keep_frames or not frame_paths:
            return

        parent = frame_paths[0].parent
        for frame_path in frame_paths:
            if frame_path.exists():
                frame_path.unlink()
        self._remove_dir_if_exists(parent)

    @staticmethod
    def _remove_dir_if_exists(path: Path) -> None:
        with suppress(OSError):
            path.rmdir()

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
