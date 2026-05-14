"""Tests for VAD implementations."""

from __future__ import annotations

import struct

import pytest

from synapsekit.voice.vad import EnergyVAD


def _make_pcm(amplitude: float, num_samples: int = 480) -> bytes:
    """Generate synthetic 16-bit PCM with constant amplitude."""
    sample = int(amplitude * 32767)
    return struct.pack(f"<{num_samples}h", *([sample] * num_samples))


def _silence(num_samples: int = 480) -> bytes:
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


class TestEnergyVAD:
    def test_silence_is_not_speech_sync(self) -> None:
        vad = EnergyVAD(threshold=0.01)
        assert vad._is_speech_sync(_silence()) is False

    def test_loud_signal_is_speech_sync(self) -> None:
        vad = EnergyVAD(threshold=0.01)
        frame = _make_pcm(amplitude=0.5)
        assert vad._is_speech_sync(frame) is True

    def test_threshold_boundary(self) -> None:
        # Amplitude 0.009 is below default threshold
        vad = EnergyVAD(threshold=0.01)
        frame = _make_pcm(amplitude=0.009)
        assert vad._is_speech_sync(frame) is False

        # Amplitude 0.02 is above threshold
        frame = _make_pcm(amplitude=0.02)
        assert vad._is_speech_sync(frame) is True

    def test_empty_frame_is_not_speech(self) -> None:
        vad = EnergyVAD(threshold=0.01)
        assert vad._is_speech_sync(b"") is False

    def test_odd_length_frame_handled(self) -> None:
        vad = EnergyVAD(threshold=0.01)
        frame = _make_pcm(amplitude=0.5) + b"\x00"  # extra byte
        # Should not raise; odd byte is truncated
        result = vad._is_speech_sync(frame)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_is_speech_async(self) -> None:
        vad = EnergyVAD(threshold=0.01)
        silence = _silence()
        loud = _make_pcm(amplitude=0.5)
        assert await vad.is_speech(silence) is False
        assert await vad.is_speech(loud) is True

    def test_custom_threshold(self) -> None:
        strict_vad = EnergyVAD(threshold=0.5)
        frame = _make_pcm(amplitude=0.3)  # above 0.01 but below 0.5
        assert strict_vad._is_speech_sync(frame) is False

        loose_vad = EnergyVAD(threshold=0.001)
        quiet = _make_pcm(amplitude=0.005)
        assert loose_vad._is_speech_sync(quiet) is True

    def test_vad_silence_does_not_trigger_stt(self) -> None:
        """VAD must suppress silence — silence should never reach STT."""
        vad = EnergyVAD(threshold=0.01)
        results = [vad._is_speech_sync(_silence()) for _ in range(20)]
        assert all(r is False for r in results)


class TestSileroVADImport:
    """SileroVAD requires torch; verify graceful ImportError without it."""

    def test_import_error_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _block_torch(name: str, *args: object, **kw: object) -> object:
            if name == "torch":
                raise ImportError("torch not available")
            return real_import(name, *args, **kw)

        monkeypatch.setattr(builtins, "__import__", _block_torch)

        from synapsekit.voice.vad import SileroVAD

        vad = SileroVAD()
        with pytest.raises(ImportError, match="torch"):
            vad._load()
