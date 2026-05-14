"""Voice Activity Detection implementations."""

from __future__ import annotations

import asyncio
import math
import struct
from typing import Any

from .base import BaseVAD


class EnergyVAD(BaseVAD):
    """
    RMS energy-based VAD for 16-bit PCM audio.

    Detects speech by comparing the root-mean-square energy of each frame
    against a normalised threshold.  Runs in a thread to avoid blocking the
    event loop.

    Suitable for quiet environments.  For production use, prefer SileroVAD.

    Parameters
    ----------
    threshold:
        Normalised RMS threshold in [0, 1].  Default 0.01 works well for
        most desktop microphones in a quiet room.
    sample_width:
        Bytes per sample (2 = 16-bit PCM, 1 = 8-bit).
    """

    def __init__(self, threshold: float = 0.01, sample_width: int = 2) -> None:
        self.threshold = threshold
        self.sample_width = sample_width

    async def is_speech(self, frame: bytes) -> bool:
        # RMS is CPU-light (<0.1 ms); running inline avoids thread-dispatch
        # overhead (~0.5-2 ms per frame) that would dwarf the computation.
        return self._is_speech_sync(frame)

    def _is_speech_sync(self, frame: bytes) -> bool:
        if not frame:
            return False
        count = len(frame) // self.sample_width
        if count == 0:
            return False
        fmt = f"<{count}h" if self.sample_width == 2 else f"<{count}b"
        try:
            samples = struct.unpack(fmt, frame[: count * self.sample_width])
        except struct.error:
            return False
        rms = math.sqrt(sum(float(s) * float(s) for s in samples) / count)
        max_val = 32768.0 if self.sample_width == 2 else 128.0
        return (rms / max_val) > self.threshold


class SileroVAD(BaseVAD):
    """
    Silero VAD — local neural Voice Activity Detector.

    Operates fully on-device with minimal CPU overhead (~2 ms/frame) and
    detects speech onset within ~100 ms.  Significantly more robust than
    energy-based approaches in noisy environments.

    Recommended for production pipelines where accuracy matters.

    Requires::

        pip install torch torchaudio  # or synapsekit[voice-silero]

    Parameters
    ----------
    threshold:
        Speech confidence threshold in [0, 1].  Default 0.5.
    sample_rate:
        Expected audio sample rate.  Silero supports 8000 and 16000 Hz.
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for SileroVAD. Install with: pip install torch torchaudio"
            ) from None
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._model = model

    async def is_speech(self, frame: bytes) -> bool:
        return await asyncio.to_thread(self._is_speech_sync, frame)

    def _is_speech_sync(self, frame: bytes) -> bool:
        self._load()
        try:
            import numpy as np
            import torch
        except ImportError:
            raise ImportError("torch and numpy are required for SileroVAD") from None

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples)
        confidence: float = self._model(tensor, self.sample_rate).item()
        return confidence >= self.threshold
