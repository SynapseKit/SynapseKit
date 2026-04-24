"""STT and TTS backends for VoiceAgent, plus simple VAD."""

from __future__ import annotations

import asyncio
import io
import math
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class STTBackend(ABC):
    """Abstract base class for Speech-to-Text backends."""

    @abstractmethod
    async def transcribe(self, audio: bytes | Path, **kwargs: Any) -> str:
        """Transcribe audio to text."""
        ...


class TTSBackend(ABC):
    """Abstract base class for Text-to-Speech backends."""

    @abstractmethod
    async def synthesize(self, text: str, **kwargs: Any) -> bytes:
        """Synthesize text to audio bytes."""
        ...


class WhisperAPIBackend(STTBackend):
    """OpenAI Whisper API STT backend."""

    def __init__(self, model: str = "whisper-1", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    async def transcribe(self, audio: bytes | Path, **kwargs: Any) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for WhisperAPIBackend. "
                "Install it with: pip install 'synapsekit[voice]'"
            ) from None

        client = openai.OpenAI(api_key=self.api_key)

        def _transcribe() -> str:
            if isinstance(audio, Path):
                with open(audio, "rb") as f:
                    transcript = client.audio.transcriptions.create(
                        model=self.model, file=f, **kwargs
                    )
            else:
                # API requires a filename tuple for raw bytes
                file_tuple = ("audio.wav", audio, "audio/wav")
                transcript = client.audio.transcriptions.create(
                    model=self.model, file=file_tuple, **kwargs
                )
            return str(getattr(transcript, "text", "") or "")

        return await asyncio.to_thread(_transcribe)


class WhisperLocalBackend(STTBackend):
    """Local Whisper STT backend (faster-whisper or openai-whisper)."""

    def __init__(self, model: str = "base") -> None:
        self.model = model

    async def transcribe(self, audio: bytes | Path, **kwargs: Any) -> str:
        def _transcribe() -> str:
            # Try faster-whisper first
            try:
                from faster_whisper import WhisperModel

                model = WhisperModel(self.model)
                if isinstance(audio, Path):
                    segments, _ = model.transcribe(str(audio), **kwargs)
                else:
                    import soundfile as sf

                    with io.BytesIO(audio) as buf:
                        data, samplerate = sf.read(buf)
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)  # to mono
                    segments, _ = model.transcribe(data, **kwargs)
                return "".join([segment.text for segment in segments])
            except ImportError:
                pass

            # Fallback to openai-whisper
            try:
                import whisper
            except ImportError:
                raise ImportError(
                    "faster-whisper or openai-whisper is required for WhisperLocalBackend. "
                    "Install with: pip install 'synapsekit[voice-local]'"
                ) from None

            model = whisper.load_model(self.model)
            if isinstance(audio, Path):
                result = model.transcribe(str(audio), **kwargs)
            else:
                import numpy as np
                import soundfile as sf

                with io.BytesIO(audio) as buf:
                    data, samplerate = sf.read(buf)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                data = data.astype(np.float32)
                result = model.transcribe(data, **kwargs)
            return str(result.get("text", ""))

        return await asyncio.to_thread(_transcribe)


class OpenAITTSBackend(TTSBackend):
    """OpenAI TTS API backend."""

    def __init__(
        self,
        model: str = "tts-1",
        voice: str = "alloy",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.api_key = api_key

    async def synthesize(self, text: str, **kwargs: Any) -> bytes:
        if not text.strip():
            return b""

        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for OpenAITTSBackend. "
                "Install it with: pip install 'synapsekit[voice]'"
            ) from None

        client = openai.OpenAI(api_key=self.api_key)

        def _synthesize() -> bytes:
            response = client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="mp3",
                **kwargs,
            )
            return response.read()

        return await asyncio.to_thread(_synthesize)


class Pyttsx3TTSBackend(TTSBackend):
    """Local pyttsx3 TTS backend."""

    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice

    async def synthesize(self, text: str, **kwargs: Any) -> bytes:
        if not text.strip():
            return b""

        try:
            import pyttsx3
        except ImportError:
            raise ImportError(
                "pyttsx3 is required for Pyttsx3TTSBackend. "
                "Install it with: pip install 'synapsekit[voice-local]'"
            ) from None

        def _synthesize() -> bytes:
            engine = pyttsx3.init()
            if self.voice:
                engine.setProperty("voice", self.voice)

            # pyttsx3 doesn't easily output to bytes directly in memory across all platforms,
            # so we save to a temp file and read it.
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            try:
                engine.save_to_file(text, temp_path)
                engine.runAndWait()
                with open(temp_path, "rb") as f2:
                    return f2.read()
            finally:
                import os

                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        return await asyncio.to_thread(_synthesize)


class EnergyVAD:
    """Simple energy-based Voice Activity Detection to filter silence."""

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def is_speech(self, audio_chunk: bytes, sample_width: int = 2) -> bool:
        """
        Check if the audio chunk contains speech based on RMS energy.
        Assumes 16-bit PCM (sample_width=2) by default.
        """
        if not audio_chunk:
            return False

        count = len(audio_chunk) // sample_width
        if count == 0:
            return False

        format_str = f"<{count}h" if sample_width == 2 else f"<{count}b"
        try:
            samples = struct.unpack(format_str, audio_chunk[: count * sample_width])
        except struct.error:
            return False

        # Calculate RMS
        sum_squares = sum(float(s) * float(s) for s in samples)
        rms = math.sqrt(sum_squares / count)

        # Normalize RMS (max for 16-bit is 32768)
        normalized_rms = rms / (32768.0 if sample_width == 2 else 128.0)

        return normalized_rms > self.threshold
