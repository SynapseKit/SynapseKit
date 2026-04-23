"""Production-grade tests for AudioLoader and VideoLoader.

Covers: _transcribe_api (mocked openai), _transcribe_local (mocked whisper),
_normalise_segments (dict + attr-based), _segment_value, _to_float,
_to_documents fallback, ffmpeg subprocess mocking, cleanup paths,
_decorate_transcript_doc, _decorate_frame_docs, _frame_timestamp.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.loaders.audio import AudioLoader
from synapsekit.loaders.base import Document
from synapsekit.loaders.video import VideoLoader

# ---------------------------------------------------------------------------
# AudioLoader — _transcribe_api
# ---------------------------------------------------------------------------


def test_transcribe_api_calls_openai(tmp_path):
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")

    mock_segment = SimpleNamespace(text="hello there", start=0.0, end=1.5)
    mock_transcript = SimpleNamespace(text="hello there", segments=[mock_segment])
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_transcript

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client

    loader = AudioLoader(str(audio_file), api_key="sk-test")
    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = loader._transcribe_api()

    assert result["text"] == "hello there"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["start"] == 0.0


def test_transcribe_api_with_language(tmp_path):
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"fake")

    mock_transcript = SimpleNamespace(text="bonjour", segments=[])
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_transcript

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client

    loader = AudioLoader(str(audio_file), api_key="sk-test", language="fr")
    with patch.dict(sys.modules, {"openai": mock_openai}):
        result = loader._transcribe_api()

    call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
    assert call_kwargs.get("language") == "fr"
    assert result["text"] == "bonjour"


def test_transcribe_api_import_error(tmp_path):
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")
    loader = AudioLoader(str(audio_file))

    with patch.dict(sys.modules, {"openai": None}):
        with pytest.raises(ImportError, match="openai is required"):
            loader._transcribe_api()


# ---------------------------------------------------------------------------
# AudioLoader — _transcribe_local
# ---------------------------------------------------------------------------


def test_transcribe_local_calls_whisper(tmp_path):
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "local transcription",
        "segments": [{"text": "local transcription", "start": 0.0, "end": 2.0}],
    }
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    loader = AudioLoader(str(audio_file), backend="whisper_local")
    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        result = loader._transcribe_local()

    assert result["text"] == "local transcription"
    assert len(result["segments"]) == 1
    mock_whisper.load_model.assert_called_once_with("base")  # "whisper-1" → "base"


def test_transcribe_local_with_custom_model(tmp_path):
    audio_file = tmp_path / "clip.flac"
    audio_file.write_bytes(b"fake")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "large result", "segments": []}
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    loader = AudioLoader(str(audio_file), backend="whisper_local", model="large")
    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        result = loader._transcribe_local()

    mock_whisper.load_model.assert_called_once_with("large")
    assert result["text"] == "large result"


def test_transcribe_local_with_language(tmp_path):
    """Line 106: language kwarg is passed to whisper.transcribe when set."""
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "spanish text", "segments": []}
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    loader = AudioLoader(str(audio_file), backend="whisper_local", language="es")
    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        loader._transcribe_local()

    call_kwargs = mock_model.transcribe.call_args[1]
    assert call_kwargs.get("language") == "es"


def test_transcribe_local_import_error(tmp_path):
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")
    loader = AudioLoader(str(audio_file), backend="whisper_local")
    with patch.dict(sys.modules, {"whisper": None}):
        with pytest.raises(ImportError, match="openai-whisper is required"):
            loader._transcribe_local()


# ---------------------------------------------------------------------------
# AudioLoader — _normalise_segments (attr-based objects)
# ---------------------------------------------------------------------------


def test_normalise_segments_with_attr_objects(tmp_path):
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")
    loader = AudioLoader(str(audio_file))

    seg1 = SimpleNamespace(text="first segment", start=0.5, end=1.5)
    seg2 = SimpleNamespace(text="", start=2.0, end=3.0)  # empty → skipped
    seg3 = SimpleNamespace(text="third", start=3.0, end=4.0)

    result = loader._normalise_segments([seg1, seg2, seg3])
    assert len(result) == 2
    assert result[0]["text"] == "first segment"
    assert result[0]["start"] == pytest.approx(0.5)
    assert result[1]["text"] == "third"


def test_normalise_segments_empty_input(tmp_path):
    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"fake")
    loader = AudioLoader(str(audio_file))
    assert loader._normalise_segments([]) == []
    assert loader._normalise_segments(None) == []


# ---------------------------------------------------------------------------
# AudioLoader — _segment_value with non-dict
# ---------------------------------------------------------------------------


def test_segment_value_attr_access():
    obj = SimpleNamespace(text="val", start=1.0)
    assert AudioLoader._segment_value(obj, "text", "default") == "val"
    assert AudioLoader._segment_value(obj, "missing_key", "fallback") == "fallback"


def test_segment_value_dict_access():
    d = {"text": "hello", "start": 2.0}
    assert AudioLoader._segment_value(d, "text", "") == "hello"
    assert AudioLoader._segment_value(d, "end", None) is None


# ---------------------------------------------------------------------------
# AudioLoader — _to_float edge cases
# ---------------------------------------------------------------------------


def test_to_float_none_returns_none():
    assert AudioLoader._to_float(None) is None


def test_to_float_valid_float():
    assert AudioLoader._to_float(3.14) == pytest.approx(3.14)
    assert AudioLoader._to_float("2.5") == pytest.approx(2.5)


def test_to_float_invalid_string_returns_none():
    assert AudioLoader._to_float("not_a_number") is None


def test_to_float_invalid_type_returns_none():
    assert AudioLoader._to_float(object()) is None


# ---------------------------------------------------------------------------
# AudioLoader — _to_documents fallback path (no segments, has text)
# ---------------------------------------------------------------------------


def test_to_documents_no_segments_uses_full_text(tmp_path):
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")
    loader = AudioLoader(str(audio_file))

    result = loader._to_documents({"text": "Plain text no segments.", "segments": []})
    assert len(result) >= 1
    assert any("Plain text" in d.text for d in result)
    assert result[0].metadata["source_type"] == "audio"


def test_to_documents_empty_segment_text_skipped(tmp_path):
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")
    loader = AudioLoader(str(audio_file))

    result = loader._to_documents(
        {
            "text": "fallback",
            "segments": [{"text": "", "start": 0.0, "end": 1.0}],
        }
    )
    # Empty segment skipped; fallback text used
    assert all("fallback" in d.text for d in result)


def test_to_documents_empty_text_and_no_segments_returns_empty(tmp_path):
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")
    loader = AudioLoader(str(audio_file))

    result = loader._to_documents({"text": "", "segments": []})
    assert result == []


# ---------------------------------------------------------------------------
# VideoLoader — _load_transcript_docs_sync/_async
# ---------------------------------------------------------------------------


def test_load_transcript_docs_sync(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    loader = VideoLoader(str(video_file))
    transcript_doc = Document(text="from audio", metadata={"start_time": 1.0})

    with patch("synapsekit.loaders.video.AudioLoader") as mock_audio_cls:
        mock_instance = MagicMock()
        mock_instance.load.return_value = [transcript_doc]
        mock_audio_cls.return_value = mock_instance
        result = loader._load_transcript_docs_sync(audio_path)

    assert len(result) == 1
    assert result[0].metadata["chunk_type"] == "transcript"
    assert result[0].metadata["source_type"] == "video"


@pytest.mark.asyncio
async def test_load_transcript_docs_async(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    loader = VideoLoader(str(video_file))
    transcript_doc = Document(text="async audio", metadata={"start_time": 2.5})

    with patch("synapsekit.loaders.video.AudioLoader") as mock_audio_cls:
        mock_instance = MagicMock()
        mock_instance.aload = AsyncMock(return_value=[transcript_doc])
        mock_audio_cls.return_value = mock_instance
        result = await loader._load_transcript_docs_async(audio_path)

    assert len(result) == 1
    assert result[0].metadata["chunk_type"] == "transcript"


# ---------------------------------------------------------------------------
# VideoLoader — _frame_documents_sync/_async
# ---------------------------------------------------------------------------


def test_frame_documents_sync(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    frame_path = tmp_path / "frame_000001.jpg"
    frame_path.write_bytes(b"img")

    loader = VideoLoader(str(video_file), frame_interval=30)
    frame_doc = Document(text="a frame", metadata={})

    with (
        patch("synapsekit.loaders.video.ImageLoader"),
        patch("synapsekit.loaders.video.run_sync", return_value=[frame_doc]),
    ):
        result = loader._frame_documents_sync([frame_path])

    assert len(result) == 1
    assert result[0].metadata["chunk_type"] == "frame_caption"
    assert result[0].metadata["frame_index"] == 1


@pytest.mark.asyncio
async def test_frame_documents_async(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    frame_path = tmp_path / "frame_000001.jpg"
    frame_path.write_bytes(b"img")

    loader = VideoLoader(str(video_file), frame_interval=30)
    frame_doc = Document(text="async frame", metadata={})

    with patch("synapsekit.loaders.video.ImageLoader") as mock_image_cls:
        mock_instance = MagicMock()
        mock_instance.aload = AsyncMock(return_value=[frame_doc])
        mock_image_cls.return_value = mock_instance
        result = await loader._frame_documents_async([frame_path])

    assert len(result) == 1
    assert result[0].metadata["chunk_type"] == "frame_caption"


# ---------------------------------------------------------------------------
# VideoLoader — _decorate_transcript_doc
# ---------------------------------------------------------------------------


def test_decorate_transcript_doc(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file))

    doc = Document(
        text="transcript",
        metadata={"start_time": 5.0, "end_time": 8.0, "source_type": "audio"},
    )
    decorated = loader._decorate_transcript_doc(doc)

    assert decorated.metadata["source"] == str(video_file)
    assert decorated.metadata["source_type"] == "video"
    assert decorated.metadata["chunk_type"] == "transcript"
    assert decorated.metadata["timestamp"] == pytest.approx(5.0)
    assert decorated.metadata["loader"] == "VideoLoader"


def test_decorate_transcript_doc_no_start_time(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file))

    doc = Document(text="no ts", metadata={})
    decorated = loader._decorate_transcript_doc(doc)
    assert decorated.metadata["timestamp"] is None


# ---------------------------------------------------------------------------
# VideoLoader — _frame_timestamp
# ---------------------------------------------------------------------------


def test_frame_timestamp_with_interval(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), frame_interval=10)

    assert loader._frame_timestamp(1) == pytest.approx(0.0)
    assert loader._frame_timestamp(2) == pytest.approx(10.0)
    assert loader._frame_timestamp(4) == pytest.approx(30.0)


def test_frame_timestamp_no_interval(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), frame_interval=None)
    assert loader._frame_timestamp(5) is None


# ---------------------------------------------------------------------------
# VideoLoader — _decorate_frame_docs
# ---------------------------------------------------------------------------


def test_decorate_frame_docs(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), frame_interval=30)

    frame_path = tmp_path / "frame_000001.jpg"
    doc = Document(text="frame caption", metadata={"source_type": "image"})

    result = loader._decorate_frame_docs([doc], frame_path, frame_index=1)
    assert result[0].metadata["chunk_type"] == "frame_caption"
    assert result[0].metadata["timestamp"] == pytest.approx(0.0)
    assert result[0].metadata["frame_index"] == 1
    assert result[0].metadata["source"] == str(video_file)


# ---------------------------------------------------------------------------
# VideoLoader — _extract_audio_sync (mocked subprocess)
# ---------------------------------------------------------------------------


def test_extract_audio_sync_success(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file))

    with patch("subprocess.run") as mock_run:
        loader._extract_audio_sync()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "ffmpeg" in cmd
    assert "-ac" in cmd


def test_extract_audio_sync_ffmpeg_not_found(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file))

    with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg")):
        with pytest.raises(RuntimeError, match="ffmpeg is required"):
            loader._extract_audio_sync()


def test_extract_audio_sync_ffmpeg_failed(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file))

    import subprocess

    err = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"error output")
    with patch("subprocess.run", side_effect=err):
        with pytest.raises(RuntimeError, match="ffmpeg failed to extract audio"):
            loader._extract_audio_sync()


# ---------------------------------------------------------------------------
# VideoLoader — _extract_frames_sync (mocked subprocess)
# ---------------------------------------------------------------------------


def test_extract_frames_sync_no_interval(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), frame_interval=None)
    result = loader._extract_frames_sync()
    assert result == []


def test_extract_frames_sync_zero_interval(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), frame_interval=0)
    result = loader._extract_frames_sync()
    assert result == []


def test_extract_frames_sync_success(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), frame_interval=30)

    # Create fake frames in a temp dir to simulate ffmpeg output
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    fake_frame = frames_dir / "frame_000001.jpg"
    fake_frame.write_bytes(b"img")

    with patch("subprocess.run"), patch("tempfile.mkdtemp", return_value=str(frames_dir)):
        result = loader._extract_frames_sync()

    assert len(result) == 1
    assert result[0].name == "frame_000001.jpg"


# ---------------------------------------------------------------------------
# VideoLoader — _make_audio_path + _cleanup_audio_file
# ---------------------------------------------------------------------------


def test_make_audio_path_keep_audio(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), keep_audio=True)
    audio_path = loader._make_audio_path()
    assert audio_path.suffix == ".wav"
    assert audio_path.stem == video_file.stem


def test_make_audio_path_temp(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), keep_audio=False)
    audio_path = loader._make_audio_path()
    assert audio_path.suffix == ".wav"
    assert "synapsekit_audio_" in audio_path.name


def test_cleanup_audio_file_skipped_when_keep_audio(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), keep_audio=True)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    loader._cleanup_audio_file(audio_path)
    assert audio_path.exists()  # not deleted


def test_cleanup_audio_file_deletes_when_not_keep(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), keep_audio=False)

    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    loader._cleanup_audio_file(audio_path)
    assert not audio_path.exists()  # deleted


def test_cleanup_frame_files_skipped_when_keep_frames(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), keep_frames=True)

    f = tmp_path / "frame_000001.jpg"
    f.write_bytes(b"img")
    loader._cleanup_frame_files([f])
    assert f.exists()


def test_cleanup_frame_files_deletes_frames(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file), keep_frames=False)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    f1 = frames_dir / "frame_000001.jpg"
    f1.write_bytes(b"img")
    loader._cleanup_frame_files([f1])
    assert not f1.exists()


def test_cleanup_frame_files_empty_list(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file))
    # Should not raise
    loader._cleanup_frame_files([])


# ---------------------------------------------------------------------------
# VideoLoader — _to_float static method
# ---------------------------------------------------------------------------


def test_video_to_float_none():
    assert VideoLoader._to_float(None) is None


def test_video_to_float_valid():
    assert VideoLoader._to_float(3) == pytest.approx(3.0)
    assert VideoLoader._to_float("1.5") == pytest.approx(1.5)


def test_video_to_float_invalid():
    assert VideoLoader._to_float("bad") is None


# ---------------------------------------------------------------------------
# VideoLoader — async extract audio path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_audio_async_success(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file))

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))
    mock_proc.returncode = 0

    with patch("synapsekit.loaders.video.asyncio.create_subprocess_exec", return_value=mock_proc):
        audio_path = await loader._extract_audio_async()

    assert audio_path.suffix == ".wav"


@pytest.mark.asyncio
async def test_extract_audio_async_ffmpeg_not_found(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file))

    with patch(
        "synapsekit.loaders.video.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("ffmpeg not found"),
    ):
        with pytest.raises(RuntimeError, match="ffmpeg is required"):
            await loader._extract_audio_async()


@pytest.mark.asyncio
async def test_extract_audio_async_ffmpeg_failed(tmp_path):
    video_file = tmp_path / "v.mp4"
    video_file.write_bytes(b"fake")
    loader = VideoLoader(str(video_file))

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"ffmpeg error"))
    mock_proc.returncode = 1

    with patch("synapsekit.loaders.video.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(RuntimeError, match="ffmpeg failed to extract audio"):
            await loader._extract_audio_async()
