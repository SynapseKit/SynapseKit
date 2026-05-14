"""Shared types for the voice pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PipelineState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class AudioFrame:
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # bytes per sample (16-bit PCM)


@dataclass
class TranscriptChunk:
    text: str
    is_final: bool = False
    confidence: float = 1.0


@dataclass
class PipelineEvent:
    """
    Event emitted by VoicePipeline throughout processing.

    ``kind`` is one of:
      - ``"state_change"`` — data is a PipelineState
      - ``"transcript"``   — data is a partial/final transcript string
      - ``"response_token"`` — data is a single LLM output token
      - ``"audio_chunk"``  — data is the byte length of the TTS chunk played
      - ``"error"``        — data is an error message string
    """

    kind: str
    data: Any = None
