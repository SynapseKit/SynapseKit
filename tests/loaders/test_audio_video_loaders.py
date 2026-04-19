"""Tests for AudioLoader and VideoLoader."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from synapsekit.loaders.audio import SUPPORTED_EXTENSIONS as AUDIO_EXTENSIONS
from synapsekit.loaders.audio import AudioLoader
from synapsekit.loaders.base import Document
from synapsekit.loaders.video import SUPPORTED_EXTENSIONS as VIDEO_EXTENSIONS
from synapsekit.loaders.video import VideoLoader


class TestAudioLoader:
    def test_supported_extensions(self):
        assert ".mp3" in AUDIO_EXTENSIONS
        assert ".wav" in AUDIO_EXTENSIONS
        assert ".m4a" in AUDIO_EXTENSIONS
        assert ".ogg" in AUDIO_EXTENSIONS
        assert ".flac" in AUDIO_EXTENSIONS

    def test_unsupported_extension_raises(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.touch()
        with pytest.raises(ValueError, match="Unsupported audio format"):
            AudioLoader(str(bad_file))

    def test_load_whisper_api_segmented(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        loader = AudioLoader(str(audio_file), api_key="sk-test", backend="whisper_api")
        fake_transcript = {
            "text": "Hello world transcription.",
            "segments": [
                {"text": "Hello world.", "start": 0.0, "end": 1.2},
                {"text": "Second sentence.", "start": 1.2, "end": 2.1},
            ],
        }
        with patch.object(loader, "_transcribe_api", return_value=fake_transcript):
            docs = loader.load()

        assert len(docs) == 2
        assert docs[0].metadata["loader"] == "AudioLoader"
        assert docs[0].metadata["source"] == str(audio_file)
        assert docs[0].metadata["source_type"] == "audio"
        assert docs[0].metadata["start_time"] == 0.0
        assert docs[0].metadata["end_time"] == 1.2

    def test_unknown_backend_raises(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"data")
        loader = AudioLoader(str(audio_file), backend="unknown")
        with pytest.raises(ValueError, match="Unknown backend"):
            loader.load()

    async def test_aload(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        loader = AudioLoader(str(audio_file), api_key="sk-test")
        with patch.object(
            loader, "_transcribe_api", return_value={"text": "Async", "segments": []}
        ):
            docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Async"

    def test_metadata_includes_backend(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"data")
        loader = AudioLoader(str(audio_file), backend="whisper_api")
        with patch.object(loader, "_transcribe_api", return_value={"text": "text", "segments": []}):
            docs = loader.load()
        assert docs[0].metadata["backend"] == "whisper_api"

    def test_whisper_local_backend(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"data")
        loader = AudioLoader(str(audio_file), backend="whisper_local")
        with patch.object(
            loader, "_transcribe_local", return_value={"text": "local text", "segments": []}
        ):
            docs = loader.load()
        assert docs[0].text == "local text"
        assert docs[0].metadata["backend"] == "whisper_local"


class TestVideoLoader:
    def test_supported_extensions(self):
        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".mov" in VIDEO_EXTENSIONS
        assert ".avi" in VIDEO_EXTENSIONS
        assert ".mkv" in VIDEO_EXTENSIONS

    def test_unsupported_extension_raises(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.touch()
        with pytest.raises(ValueError, match="Unsupported video format"):
            VideoLoader(str(bad_file))

    def test_load_includes_transcript_and_frames(self, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        audio_file = tmp_path / "extracted.wav"
        audio_file.write_bytes(b"fake audio")

        frame1 = tmp_path / "frame_000001.jpg"
        frame1.write_bytes(b"frame1")
        frame2 = tmp_path / "frame_000002.jpg"
        frame2.write_bytes(b"frame2")

        loader = VideoLoader(str(video_file), api_key="sk-test", frame_interval=30)

        transcript_doc = Document(
            text="Video transcription",
            metadata={"start_time": 5.0, "end_time": 9.5, "source_type": "audio"},
        )
        frame_doc = Document(text="A UI screen", metadata={"source_type": "image"})

        with patch.object(loader, "_extract_audio_sync", return_value=audio_file):
            with patch.object(loader, "_extract_frames_sync", return_value=[frame1, frame2]):
                with patch.object(
                    loader, "_load_transcript_docs_sync", return_value=[transcript_doc]
                ):
                    with patch.object(
                        loader,
                        "_frame_documents_sync",
                        return_value=[frame_doc],
                    ):
                        docs = loader.load()

        assert len(docs) == 2
        assert docs[0].text == "Video transcription"
        assert docs[1].text == "A UI screen"

    def test_keep_audio_flag(self, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")
        loader = VideoLoader(str(video_file), keep_audio=True)
        assert loader._keep_audio is True

    async def test_aload(self, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        audio_file = tmp_path / "extracted.wav"
        audio_file.write_bytes(b"fake audio")

        loader = VideoLoader(str(video_file), api_key="sk-test")
        transcript_doc = Document(text="Async video", metadata={"start_time": 0.0})

        with patch.object(loader, "_extract_audio_async", return_value=audio_file):
            with patch.object(loader, "_extract_frames_async", return_value=[]):
                with patch.object(
                    loader,
                    "_load_transcript_docs_async",
                    new=AsyncMock(return_value=[transcript_doc]),
                ):
                    with patch.object(
                        loader,
                        "_frame_documents_async",
                        new=AsyncMock(return_value=[]),
                    ):
                        docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Async video"

    def test_webm_shared_extension(self, tmp_path):
        """`.webm` is valid for both audio and video."""
        webm_file = tmp_path / "test.webm"
        webm_file.write_bytes(b"data")
        assert ".webm" in AUDIO_EXTENSIONS
        assert ".webm" in VIDEO_EXTENSIONS
