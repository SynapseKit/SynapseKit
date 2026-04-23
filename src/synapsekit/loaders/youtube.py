from __future__ import annotations

import asyncio
import re
from urllib.parse import parse_qs, urlparse

from .base import Document

# Matches a bare YouTube video ID: 11 alphanumeric/dash/underscore chars
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _extract_video_id(url_or_id: str) -> str:
    """Return the video ID embedded in *url_or_id*.

    Accepts:
    * ``https://www.youtube.com/watch?v=VIDEO_ID``
    * ``https://youtu.be/VIDEO_ID``
    * A bare 11-character video ID
    """
    parsed = urlparse(url_or_id)

    # youtu.be/<id>
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(video_id):
            return video_id

    # youtube.com/watch?v=<id>
    if parsed.netloc in ("youtube.com", "www.youtube.com"):
        qs = parse_qs(parsed.query)
        ids = qs.get("v", [])
        if ids and _VIDEO_ID_RE.match(ids[0]):
            return ids[0]

    # Bare video ID
    if _VIDEO_ID_RE.match(url_or_id):
        return url_or_id

    raise ValueError(f"Could not extract a valid YouTube video ID from: {url_or_id!r}")


class YouTubeLoader:
    """Load a YouTube video transcript as a :class:`Document`.

    Parameters
    ----------
    url_or_id : str
        A full YouTube URL or a bare 11-character video ID.
    language : str | None
        BCP-47 language code (e.g. ``"en"``) for the preferred transcript
        language.  When ``None`` the API returns its default language.
    """

    def __init__(self, url_or_id: str, language: str | None = None) -> None:
        if not url_or_id:
            raise ValueError("url_or_id must be provided")
        self._video_id = _extract_video_id(url_or_id)
        self._language = language

    def load(self) -> list[Document]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import (
                NoTranscriptFound,
                TranscriptsDisabled,
                VideoUnavailable,
            )
        except ImportError:
            raise ImportError(
                "youtube-transcript-api required: pip install synapsekit[youtube]"
            ) from None

        ytt = YouTubeTranscriptApi()
        try:
            if self._language:
                transcript = ytt.fetch(self._video_id, languages=[self._language])
            else:
                transcript = ytt.fetch(self._video_id)
        # Known transcript-related failures are treated as "no data"
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
            return []
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch transcript: {exc}") from exc

        text = " ".join(snippet.text for snippet in transcript)
        if not text:
            return []

        return [
            Document(
                text=text,
                metadata={
                    "source": "youtube",
                    "video_id": self._video_id,
                    "language": getattr(transcript, "language", None),
                    "language_code": getattr(transcript, "language_code", None),
                },
            )
        ]

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
