"""Tests for AudioLoader and VideoLoader (v1.3.0)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from synapsekit.loaders.audio import SUPPORTED_EXTENSIONS as AUDIO_EXTENSIONS
from synapsekit.loaders.audio import AudioLoader
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

    def test_load_whisper_api(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        loader = AudioLoader(str(audio_file), api_key="sk-test", backend="whisper_api")
        with patch.object(loader, "_transcribe_api", return_value="Hello world transcription"):
            docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Hello world transcription"
        assert docs[0].metadata["loader"] == "AudioLoader"
        assert docs[0].metadata["source"] == str(audio_file)

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
        with patch.object(loader, "_transcribe_api", return_value="Async transcription"):
            docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Async transcription"

    def test_metadata_includes_backend(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"data")
        loader = AudioLoader(str(audio_file), backend="whisper_api")
        with patch.object(loader, "_transcribe_api", return_value="text"):
            docs = loader.load()
        assert docs[0].metadata["backend"] == "whisper_api"

    def test_whisper_local_backend(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"data")
        loader = AudioLoader(str(audio_file), backend="whisper_local")
        with patch.object(loader, "_transcribe_local", return_value="local text"):
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

    def test_load_delegates_to_audio_loader(self, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video")

        audio_file = tmp_path / "extracted.wav"
        audio_file.write_bytes(b"fake audio")

        loader = VideoLoader(str(video_file), api_key="sk-test")
        with patch.object(loader, "_extract_audio_sync", return_value=audio_file):
            with patch.object(AudioLoader, "_transcribe_api", return_value="Video transcription"):
                docs = loader.load()

        assert len(docs) == 1
        assert docs[0].text == "Video transcription"
        assert docs[0].metadata["loader"] == "VideoLoader"
        assert docs[0].metadata["source"] == str(video_file)

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
        with patch.object(loader, "_extract_audio_async", return_value=audio_file):
            with patch.object(AudioLoader, "_transcribe_api", return_value="Async video"):
                docs = await loader.aload()

        assert len(docs) == 1
        assert docs[0].text == "Async video"

    def test_webm_shared_extension(self, tmp_path):
        """`.webm` is valid for both audio and video."""
        webm_file = tmp_path / "test.webm"
        webm_file.write_bytes(b"data")
        # Should be valid for both
        assert ".webm" in AUDIO_EXTENSIONS
        assert ".webm" in VIDEO_EXTENSIONS
