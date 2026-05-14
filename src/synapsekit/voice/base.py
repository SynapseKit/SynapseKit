"""Abstract base classes for VAD, STT, and TTS providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseVAD(ABC):
    """Abstract Voice Activity Detector."""

    @abstractmethod
    async def is_speech(self, frame: bytes) -> bool:
        """Return True if *frame* contains speech."""
        ...


class BaseSTT(ABC):
    """
    Abstract Speech-to-Text provider with streaming support.

    Concrete implementations should yield incremental transcript strings
    as audio arrives.  The final transcript is the concatenation of all
    yielded chunks.
    """

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        """
        Yield partial transcript strings as audio is consumed from *audio_stream*.

        The caller must fully consume the returned iterator; the underlying
        audio_stream is drained as a side-effect.
        """
        # Satisfy type checker — implementations use ``yield``
        yield ""  # pragma: no cover

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe a single complete audio buffer."""

        async def _source() -> AsyncIterator[bytes]:
            yield audio

        parts: list[str] = []
        async for chunk in self.transcribe_stream(_source()):
            parts.append(chunk)
        return "".join(parts)


class BaseTTS(ABC):
    """
    Abstract Text-to-Speech provider with streaming support.

    Concrete implementations consume a streaming text iterator and yield
    audio byte chunks (raw PCM or encoded audio) as soon as each sentence
    is ready — playback can start before the full response is synthesised.
    """

    @abstractmethod
    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """
        Yield audio byte chunks from *text_stream*.

        Implementations should buffer text until a sentence boundary is
        reached, synthesise that sentence, and yield the audio immediately,
        enabling sub-500 ms perceived latency on the first sentence.
        """
        yield b""  # pragma: no cover

    async def synthesize(self, text: str) -> bytes:
        """Synthesise a complete string and return all audio bytes."""

        async def _source() -> AsyncIterator[str]:
            yield text

        chunks: list[bytes] = []
        async for chunk in self.synthesize_stream(_source()):
            chunks.append(chunk)
        return b"".join(chunks)
