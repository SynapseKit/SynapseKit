from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.loaders import Document
from synapsekit.loaders.youtube import YouTubeLoader, _extract_video_id

FAKE_ID = "dQw4w9WgXcQ"

# ---------------------------------------------------------------------------
# _extract_video_id unit tests (no network)
# ---------------------------------------------------------------------------


def test_extract_bare_id() -> None:
    assert _extract_video_id(FAKE_ID) == FAKE_ID


def test_extract_full_url() -> None:
    assert _extract_video_id(f"https://www.youtube.com/watch?v={FAKE_ID}") == FAKE_ID


def test_extract_short_url() -> None:
    assert _extract_video_id(f"https://youtu.be/{FAKE_ID}") == FAKE_ID


def test_extract_url_with_extra_params() -> None:
    url = f"https://www.youtube.com/watch?v={FAKE_ID}&t=42s&list=PL123"
    assert _extract_video_id(url) == FAKE_ID


def test_extract_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Could not extract"):
        _extract_video_id("not-a-valid-id")


def test_extract_empty_raises() -> None:
    with pytest.raises(ValueError, match="url_or_id must be provided"):
        YouTubeLoader(url_or_id="")


# ---------------------------------------------------------------------------
# Helpers — build mock transcript objects
# ---------------------------------------------------------------------------


def _make_snippet(text: str, start: float = 0.0, duration: float = 2.0) -> MagicMock:
    s = MagicMock()
    s.text = text
    s.start = start
    s.duration = duration
    return s


def _make_transcript(
    snippets: list[MagicMock], lang: str = "English", code: str = "en"
) -> MagicMock:
    t = MagicMock()
    t.language = lang
    t.language_code = code
    t.__iter__ = MagicMock(return_value=iter(snippets))
    return t


def _make_api(transcript: MagicMock) -> MagicMock:
    api = MagicMock()
    api.fetch.return_value = transcript
    return api


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------


def test_load_import_error_missing_library() -> None:
    with patch.dict("sys.modules", {"youtube_transcript_api": None}):
        loader = YouTubeLoader(FAKE_ID)
        with pytest.raises(ImportError, match="youtube-transcript-api required"):
            loader.load()


# ---------------------------------------------------------------------------
# Normal load
# ---------------------------------------------------------------------------


def test_load_returns_single_document() -> None:
    snippets = [_make_snippet("Hello"), _make_snippet("World")]
    transcript = _make_transcript(snippets)
    api_instance = _make_api(transcript)

    mock_lib = MagicMock()
    mock_lib.YouTubeTranscriptApi.return_value = api_instance
    mock_lib._errors.TranscriptsDisabled = Exception
    mock_lib._errors.NoTranscriptFound = Exception
    mock_lib._errors.VideoUnavailable = Exception

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        loader = YouTubeLoader(FAKE_ID)
        docs = loader.load()

    assert len(docs) == 1
    assert isinstance(docs[0], Document)


def test_load_text_is_joined() -> None:
    snippets = [_make_snippet("Part one"), _make_snippet("Part two")]
    transcript = _make_transcript(snippets)
    api_instance = _make_api(transcript)

    mock_lib = MagicMock()
    mock_lib.YouTubeTranscriptApi.return_value = api_instance
    mock_lib._errors.TranscriptsDisabled = Exception
    mock_lib._errors.NoTranscriptFound = Exception
    mock_lib._errors.VideoUnavailable = Exception

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        docs = YouTubeLoader(FAKE_ID).load()

    assert docs[0].text == "Part one Part two"


def test_load_metadata_correctness() -> None:
    snippets = [_make_snippet("Hello world")]
    transcript = _make_transcript(snippets, lang="English", code="en")
    api_instance = _make_api(transcript)

    mock_lib = MagicMock()
    mock_lib.YouTubeTranscriptApi.return_value = api_instance
    mock_lib._errors.TranscriptsDisabled = Exception
    mock_lib._errors.NoTranscriptFound = Exception
    mock_lib._errors.VideoUnavailable = Exception

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        docs = YouTubeLoader(FAKE_ID).load()

    meta = docs[0].metadata
    assert meta["source"] == "youtube"
    assert meta["video_id"] == FAKE_ID
    assert meta["language"] == "English"
    assert meta["language_code"] == "en"


def test_load_with_language_passes_languages_arg() -> None:
    snippets = [_make_snippet("Hola")]
    transcript = _make_transcript(snippets, lang="Spanish", code="es")
    api_instance = _make_api(transcript)

    mock_lib = MagicMock()
    mock_lib.YouTubeTranscriptApi.return_value = api_instance
    mock_lib._errors.TranscriptsDisabled = Exception
    mock_lib._errors.NoTranscriptFound = Exception
    mock_lib._errors.VideoUnavailable = Exception

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        YouTubeLoader(FAKE_ID, language="es").load()

    api_instance.fetch.assert_called_once_with(FAKE_ID, languages=["es"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_load_empty_transcript_returns_empty() -> None:
    transcript = _make_transcript([])
    transcript.__iter__ = MagicMock(return_value=iter([]))
    api_instance = _make_api(transcript)

    mock_lib = MagicMock()
    mock_lib.YouTubeTranscriptApi.return_value = api_instance
    mock_lib._errors.TranscriptsDisabled = Exception
    mock_lib._errors.NoTranscriptFound = Exception
    mock_lib._errors.VideoUnavailable = Exception

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        docs = YouTubeLoader(FAKE_ID).load()

    assert docs == []


# ---------------------------------------------------------------------------
# API error handling
# ---------------------------------------------------------------------------


def _patched_lib_with_errors() -> tuple[MagicMock, type, type, type]:
    """Return (mock_lib, TranscriptsDisabledError, NoTranscriptFoundError, VideoUnavailableError)."""

    class TranscriptsDisabledError(Exception): ...

    class NoTranscriptFoundError(Exception): ...

    class VideoUnavailableError(Exception): ...

    errors_mod = MagicMock()
    errors_mod.TranscriptsDisabled = TranscriptsDisabledError
    errors_mod.NoTranscriptFound = NoTranscriptFoundError
    errors_mod.VideoUnavailable = VideoUnavailableError

    mock_lib = MagicMock()
    mock_lib._errors = errors_mod

    return mock_lib, TranscriptsDisabledError, NoTranscriptFoundError, VideoUnavailableError


def test_transcripts_disabled_returns_empty() -> None:
    mock_lib, transcripts_disabled_error, _, _ = _patched_lib_with_errors()
    api = MagicMock()
    api.fetch.side_effect = transcripts_disabled_error()
    mock_lib.YouTubeTranscriptApi.return_value = api

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        docs = YouTubeLoader(FAKE_ID).load()

    assert docs == []


def test_no_transcript_found_returns_empty() -> None:
    mock_lib, _, no_transcript_found_error, _ = _patched_lib_with_errors()
    api = MagicMock()
    api.fetch.side_effect = no_transcript_found_error()
    mock_lib.YouTubeTranscriptApi.return_value = api

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        docs = YouTubeLoader(FAKE_ID).load()

    assert docs == []


def test_video_unavailable_returns_empty() -> None:
    mock_lib, _, _, video_unavailable_error = _patched_lib_with_errors()
    api = MagicMock()
    api.fetch.side_effect = video_unavailable_error()
    mock_lib.YouTubeTranscriptApi.return_value = api

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        docs = YouTubeLoader(FAKE_ID).load()

    assert docs == []


def test_unexpected_error_raises_runtime_error() -> None:
    mock_lib, _, _, _ = _patched_lib_with_errors()
    api = MagicMock()
    api.fetch.side_effect = ConnectionError("timeout")
    mock_lib.YouTubeTranscriptApi.return_value = api

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        with pytest.raises(RuntimeError, match="Failed to fetch transcript"):
            YouTubeLoader(FAKE_ID).load()


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_aload_returns_documents() -> None:
    snippets = [_make_snippet("Async test")]
    transcript = _make_transcript(snippets)
    api_instance = _make_api(transcript)

    mock_lib = MagicMock()
    mock_lib.YouTubeTranscriptApi.return_value = api_instance
    mock_lib._errors.TranscriptsDisabled = Exception
    mock_lib._errors.NoTranscriptFound = Exception
    mock_lib._errors.VideoUnavailable = Exception

    with patch.dict(
        "sys.modules",
        {"youtube_transcript_api": mock_lib, "youtube_transcript_api._errors": mock_lib._errors},
    ):
        docs = asyncio.run(YouTubeLoader(FAKE_ID).aload())

    assert len(docs) == 1
    assert docs[0].text == "Async test"
