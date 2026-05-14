"""Speech-to-Text provider implementations."""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator
from typing import Any

from .base import BaseSTT


def _pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


class LocalWhisperSTT(BaseSTT):
    """
    Local Whisper STT — the preferred provider for latency-sensitive desktop assistants.

    Runs fully on-device with no network round-trips, enabling low-latency
    interactions typical of desktop voice agents.  Typical transcription time
    is 100-300 ms for short utterances on modern CPUs.

    Supported backends (tried in order):

    1. **faster-whisper** — quantised model via CTranslate2 (int8/float16).
       Uses CTranslate2 model format; not compatible with whisper.cpp GGML
       files or WhisperKit binaries.
       Install: ``pip install faster-whisper``

    2. **openai-whisper** — reference PyTorch implementation.
       Install: ``pip install openai-whisper``

    Parameters
    ----------
    model:
        Whisper model size: ``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, or ``"large-v3"``.  Default ``"base"``.
    language:
        ISO-639-1 language code.  ``None`` enables auto-detection.
    sample_rate:
        Audio sample rate of the incoming PCM stream.  Default 16000.
    """

    def __init__(
        self,
        model: str = "base",
        language: str | None = None,
        sample_rate: int = 16000,
    ) -> None:
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self._loaded_model: Any = None
        self._backend: str | None = None  # "faster_whisper" or "whisper"

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        """
        Buffer all audio from *audio_stream*, then transcribe and yield the result.

        Whisper is a sequence-to-sequence model that requires the full audio
        context for high accuracy, so partial yielding happens after the stream
        is exhausted.  Use DeepgramSTT for true sub-utterance partial results.
        """
        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)

        if not buffer:
            return

        text = await asyncio.to_thread(self._transcribe_pcm, bytes(buffer))
        if text:
            yield text

    def _load_model(self) -> None:
        """Lazy-load the Whisper model once and cache it on the instance."""
        if self._loaded_model is not None:
            return

        try:
            from faster_whisper import WhisperModel

            self._loaded_model = WhisperModel(self.model, compute_type="int8")
            self._backend = "faster_whisper"
            return
        except ImportError:
            pass

        try:
            import whisper

            self._loaded_model = whisper.load_model(self.model)
            self._backend = "whisper"
            return
        except ImportError:
            pass

        raise ImportError(
            "faster-whisper or openai-whisper is required for LocalWhisperSTT. "
            "Install with: pip install faster-whisper  "
            "or: pip install 'synapsekit[voice-local]'"
        )

    def _transcribe_pcm(self, pcm: bytes) -> str:
        self._load_model()
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        kw: dict[str, Any] = {}
        if self.language:
            kw["language"] = self.language

        if self._backend == "faster_whisper":
            segments, _ = self._loaded_model.transcribe(samples, **kw)
            return "".join(seg.text for seg in segments).strip()

        # openai-whisper
        result = self._loaded_model.transcribe(samples, **kw)
        return str(result.get("text", "")).strip()


class OpenAIWhisperSTT(BaseSTT):
    """
    OpenAI Whisper API STT.

    High quality but incurs a network round-trip (typically 500 ms-1 s).
    For latency-sensitive desktop pipelines, prefer LocalWhisperSTT.

    Parameters
    ----------
    model:
        Whisper model identifier.  Default ``"whisper-1"``.
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var if omitted.
    language:
        ISO-639-1 language code for forced language detection.
    sample_rate:
        Sample rate of incoming PCM audio.  Used when encoding to WAV.
    """

    def __init__(
        self,
        model: str = "whisper-1",
        api_key: str | None = None,
        language: str | None = None,
        sample_rate: int = 16000,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.language = language
        self.sample_rate = sample_rate
        self._client: Any = None

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)

        if not buffer:
            return

        text = await asyncio.to_thread(self._transcribe_api, bytes(buffer))
        if text:
            yield text

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAIWhisperSTT. "
                    "Install with: pip install 'synapsekit[voice]'"
                ) from None
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def _transcribe_api(self, pcm: bytes) -> str:
        client = self._get_client()
        wav_bytes = _pcm_to_wav(pcm, self.sample_rate)
        kw: dict[str, Any] = {}
        if self.language:
            kw["language"] = self.language
        transcript = client.audio.transcriptions.create(
            model=self.model,
            file=("audio.wav", wav_bytes, "audio/wav"),
            **kw,
        )
        return str(getattr(transcript, "text", "") or "").strip()


class DeepgramSTT(BaseSTT):
    """
    Deepgram real-time streaming STT.

    Yields partial (interim) and final transcripts incrementally as audio
    arrives over a WebSocket, enabling near-zero-latency partial results.

    Requires::

        pip install deepgram-sdk  # or synapsekit[voice-deepgram]

    Parameters
    ----------
    api_key:
        Deepgram API key.
    model:
        Deepgram model name.  Default ``"nova-2"``.
    language:
        BCP-47 language code.  Default ``"en"``.
    sample_rate:
        Audio sample rate in Hz.  Default 16000.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        language: str = "en",
        sample_rate: int = 16000,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from deepgram import DeepgramClient
            except ImportError:
                raise ImportError(
                    "deepgram-sdk is required for DeepgramSTT. "
                    "Install with: pip install deepgram-sdk"
                ) from None
            self._client = DeepgramClient(self.api_key)
        return self._client

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        try:
            from deepgram import LiveOptions, LiveTranscriptionEvents
        except ImportError:
            raise ImportError(
                "deepgram-sdk is required for DeepgramSTT. Install with: pip install deepgram-sdk"
            ) from None

        client = self._get_client()
        transcript_queue: asyncio.Queue[str | None] = asyncio.Queue()

        connection = await client.listen.asynclive.v("1")

        async def _on_message(self_ref: Any, result: Any, **_kw: Any) -> None:
            text = result.channel.alternatives[0].transcript
            if text:
                await transcript_queue.put(text)

        connection.on(LiveTranscriptionEvents.Transcript, _on_message)

        options = LiveOptions(
            model=self.model,
            language=self.language,
            sample_rate=self.sample_rate,
            channels=1,
            encoding="linear16",
            interim_results=True,
        )
        await connection.start(options)

        async def _send() -> None:
            try:
                async for chunk in audio_stream:
                    await connection.send(chunk)
            finally:
                await connection.finish()
                await transcript_queue.put(None)

        send_task = asyncio.create_task(_send())
        try:
            while True:
                item = await transcript_queue.get()
                if item is None:
                    break
                yield item
        finally:
            send_task.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await send_task
