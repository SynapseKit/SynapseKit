"""Text-to-Speech provider implementations with sentence-level streaming."""

from __future__ import annotations

import asyncio
import io
import re
import wave
from collections.abc import AsyncIterator
from typing import Any

from .base import BaseTTS

# Sentence boundary: end punctuation followed by whitespace or end-of-string
_SENTENCE_RE = re.compile(r"(?<=[.!?:])\s+")
_MAX_SENTENCE_CHARS = 300


def _split_sentences(text: str) -> list[str]:
    """
    Split *text* at sentence boundaries.

    The last element may be an incomplete sentence — callers should hold it
    in a buffer and only synthesise it once the stream is fully consumed or
    after the next sentence boundary arrives.
    """
    parts = _SENTENCE_RE.split(text)
    # Also split on very long chunks to avoid long TTS calls
    result: list[str] = []
    for part in parts:
        while len(part) > _MAX_SENTENCE_CHARS:
            # Hard split at last space before limit
            cut = part.rfind(" ", 0, _MAX_SENTENCE_CHARS)
            cut = cut if cut > 0 else _MAX_SENTENCE_CHARS
            result.append(part[:cut])
            part = part[cut:].lstrip()
        result.append(part)
    return result or [text]


class OpenAITTS(BaseTTS):
    """
    OpenAI TTS API with sentence-level streaming.

    Synthesises text sentence by sentence so the first audio chunk is ready
    and playing before the full LLM response has been generated, achieving
    sub-500 ms perceived latency on the first sentence.

    Output format is raw 24 kHz 16-bit PCM (``response_format="pcm"``) by
    default, which the pipeline's AudioPlayer can play without decoding.

    Parameters
    ----------
    model:
        ``"tts-1"`` (low latency) or ``"tts-1-hd"`` (higher quality).
    voice:
        One of ``"alloy"``, ``"echo"``, ``"fable"``, ``"onyx"``,
        ``"nova"``, ``"shimmer"``.
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    response_format:
        Audio format returned by the API.  ``"pcm"`` gives raw 24 kHz
        16-bit mono PCM ready for sounddevice; ``"mp3"`` gives a smaller
        file suitable for storage.
    """

    def __init__(
        self,
        model: str = "tts-1",
        voice: str = "alloy",
        api_key: str | None = None,
        response_format: str = "pcm",
    ) -> None:
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.response_format = response_format
        self._client: Any = None

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        buffer = ""
        async for token in text_stream:
            buffer += token
            sentences = _split_sentences(buffer)
            # All but the last are complete sentences
            for sentence in sentences[:-1]:
                if sentence.strip():
                    audio = await asyncio.to_thread(self._synthesize_text, sentence.strip())
                    if audio:
                        yield audio
            buffer = sentences[-1] if sentences else ""

        # Flush remaining buffer
        if buffer.strip():
            audio = await asyncio.to_thread(self._synthesize_text, buffer.strip())
            if audio:
                yield audio

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAITTS. "
                    "Install with: pip install 'synapsekit[voice]'"
                ) from None
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def _synthesize_text(self, text: str) -> bytes:
        import typing

        client = self._get_client()
        response = client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format=self.response_format,
        )
        return typing.cast(bytes, response.read())


class ElevenLabsTTS(BaseTTS):
    """
    ElevenLabs streaming TTS.

    Streams audio chunks as they are generated, enabling very low first-chunk
    latency (~150 ms typical on turbo models).

    Requires::

        pip install elevenlabs  # or synapsekit[voice-elevenlabs]

    Parameters
    ----------
    api_key:
        ElevenLabs API key.
    voice_id:
        ElevenLabs voice identifier.  Default is the ``"Rachel"`` voice.
    model_id:
        Model to use.  ``"eleven_turbo_v2"`` offers the best latency.
    output_format:
        Audio format.  Default ``"pcm_24000"`` for raw 24 kHz PCM.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_turbo_v2",
        output_format: str = "pcm_24000",
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from elevenlabs.client import AsyncElevenLabs
            except ImportError:
                raise ImportError(
                    "elevenlabs is required for ElevenLabsTTS. Install with: pip install elevenlabs"
                ) from None
            self._client = AsyncElevenLabs(api_key=self.api_key)
        return self._client

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        # Sentence-level streaming: synthesise each sentence as it arrives
        # so the first audio chunk plays before the full LLM response is done.
        buffer = ""
        async for token in text_stream:
            buffer += token
            sentences = _split_sentences(buffer)
            for sentence in sentences[:-1]:
                if sentence.strip():
                    async for chunk in self._stream_sentence(sentence.strip()):
                        yield chunk
            buffer = sentences[-1] if sentences else ""

        if buffer.strip():
            async for chunk in self._stream_sentence(buffer.strip()):
                yield chunk

    async def _stream_sentence(self, text: str) -> AsyncIterator[bytes]:
        client = self._get_client()
        audio_stream = client.generate(
            text=text,
            voice=self.voice_id,
            model=self.model_id,
            output_format=self.output_format,
            stream=True,
        )
        async for chunk in await audio_stream:
            if chunk:
                yield chunk


class CartesiaTTS(BaseTTS):
    """
    Cartesia streaming TTS.

    Streams PCM audio chunks as they are generated using Cartesia's SSE API,
    enabling very low first-chunk latency (~80-150 ms typical on Sonic models).
    Each sentence is synthesised independently so playback starts before the
    full LLM response is ready.

    Output is raw signed 16-bit PCM at ``sample_rate`` Hz (default 24000),
    ready for ``_AudioPlayer`` without decoding.

    Requires::

        pip install cartesia  # or synapsekit[voice-cartesia]

    Parameters
    ----------
    api_key:
        Cartesia API key.
    voice_id:
        Cartesia voice UUID.
    model_id:
        Model to use.  ``"sonic-2"`` is the recommended low-latency model.
    language:
        BCP-47 language code.  Default ``"en"``.
    sample_rate:
        PCM output sample rate in Hz.  Default 24000.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091",
        model_id: str = "sonic-2",
        language: str = "en",
        sample_rate: int = 24000,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.language = language
        self.sample_rate = sample_rate
        self._client: Any = None

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        buffer = ""
        async for token in text_stream:
            buffer += token
            sentences = _split_sentences(buffer)
            for sentence in sentences[:-1]:
                if sentence.strip():
                    async for chunk in self._stream_sentence(sentence.strip()):
                        yield chunk
            buffer = sentences[-1] if sentences else ""

        if buffer.strip():
            async for chunk in self._stream_sentence(buffer.strip()):
                yield chunk

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from cartesia import AsyncCartesia
            except ImportError:
                raise ImportError(
                    "cartesia is required for CartesiaTTS. "
                    "Install with: pip install 'synapsekit[voice-cartesia]'"
                ) from None
            # Reuse the client across sentences to share the HTTP connection pool
            # and avoid per-sentence TLS handshake overhead (~100 ms each).
            self._client = AsyncCartesia(api_key=self.api_key)
        return self._client

    async def _stream_sentence(self, text: str) -> AsyncIterator[bytes]:
        client = self._get_client()
        async for output in client.tts.sse(
            model_id=self.model_id,
            transcript=text,
            voice={"mode": "id", "id": self.voice_id},
            language=self.language,
            output_format={
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
        ):
            if output.audio:
                yield output.audio


class PiperTTS(BaseTTS):
    """
    Piper — fully local neural TTS.

    Runs entirely on-device with very low latency (~50 ms) and no cloud
    dependency.  Suitable for privacy-sensitive or offline desktop agents.

    Requires::

        pip install piper-tts

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` Piper model file.
    config_path:
        Path to the model JSON config.  Inferred from ``model_path`` if omitted.
    speaker:
        Speaker ID for multi-speaker models.
    sample_rate:
        Expected output sample rate (model-dependent, typically 16000 or 22050).
    """

    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        speaker: int = 0,
        sample_rate: int = 22050,
    ) -> None:
        self.model_path = model_path
        self.config_path = config_path
        self.speaker = speaker
        self.sample_rate = sample_rate
        self._voice: Any = None

    def _get_voice(self) -> Any:
        if self._voice is None:
            try:
                from piper import PiperVoice
            except ImportError:
                raise ImportError(
                    "piper-tts is required for PiperTTS. "
                    "Install with: pip install 'synapsekit[voice-piper]'"
                ) from None
            self._voice = PiperVoice.load(self.model_path, config_path=self.config_path)
        return self._voice

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        buffer = ""
        async for token in text_stream:
            buffer += token
            sentences = _split_sentences(buffer)
            for sentence in sentences[:-1]:
                if sentence.strip():
                    audio = await asyncio.to_thread(self._synthesize_text, sentence.strip())
                    if audio:
                        yield audio
            buffer = sentences[-1] if sentences else ""

        if buffer.strip():
            audio = await asyncio.to_thread(self._synthesize_text, buffer.strip())
            if audio:
                yield audio

    def _synthesize_text(self, text: str) -> bytes:
        voice = self._get_voice()
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            voice.synthesize(text, wf, speaker_id=self.speaker)
        return wav_io.getvalue()
