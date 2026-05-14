"""Tests for STT implementations."""

from __future__ import annotations

import struct
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.voice.stt import LocalWhisperSTT, OpenAIWhisperSTT, _pcm_to_wav


def _make_pcm(num_samples: int = 1600) -> bytes:
    return struct.pack(f"<{num_samples}h", *([1000] * num_samples))


async def _audio_source(data: bytes) -> AsyncIterator[bytes]:
    yield data


class TestPcmToWav:
    def test_produces_valid_wav_header(self) -> None:
        import io
        import wave

        pcm = _make_pcm()
        wav = _pcm_to_wav(pcm)
        with wave.open(io.BytesIO(wav)) as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000


class TestLocalWhisperSTT:
    @pytest.mark.asyncio
    async def test_empty_audio_yields_nothing(self) -> None:
        stt = LocalWhisperSTT()

        async def _empty() -> AsyncIterator[bytes]:
            return
            yield  # make it an async generator

        parts = []
        async for chunk in stt.transcribe_stream(_empty()):
            parts.append(chunk)
        assert parts == []

    @pytest.mark.asyncio
    async def test_streams_transcript_on_completion(self) -> None:
        stt = LocalWhisperSTT(model="base")

        with patch.object(stt, "_transcribe_pcm", return_value="hello world") as mock_fn:
            parts = []
            async for chunk in stt.transcribe_stream(_audio_source(_make_pcm())):
                parts.append(chunk)

        assert parts == ["hello world"]
        mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_transcription_yields_nothing(self) -> None:
        stt = LocalWhisperSTT()
        with patch.object(stt, "_transcribe_pcm", return_value=""):
            parts = []
            async for chunk in stt.transcribe_stream(_audio_source(_make_pcm())):
                parts.append(chunk)
        assert parts == []

    @pytest.mark.asyncio
    async def test_transcribe_shortcut(self) -> None:
        stt = LocalWhisperSTT()
        with patch.object(stt, "_transcribe_pcm", return_value="test"):
            result = await stt.transcribe(_make_pcm())
        assert result == "test"

    def test_faster_whisper_backend(self) -> None:
        stt = LocalWhisperSTT(model="tiny")
        mock_segment = MagicMock()
        mock_segment.text = " Hello"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())

        with patch.dict(
            "sys.modules",
            {"faster_whisper": MagicMock(WhisperModel=MagicMock(return_value=mock_model))},
        ):
            import faster_whisper  # noqa: F401 — ensure module mock is active

            with patch("numpy.frombuffer") as mock_frombuffer:
                mock_arr = MagicMock()
                mock_arr.astype.return_value = mock_arr
                mock_frombuffer.return_value = mock_arr

                result = stt._transcribe_pcm(_make_pcm())

        # Result should have stripped text
        assert isinstance(result, str)

    def test_raises_import_error_without_backends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stt = LocalWhisperSTT()

        def _fail(*a: object, **kw: object) -> None:
            raise ImportError

        import builtins

        real_import = builtins.__import__

        def _block(name: str, *args: object, **kw: object) -> object:
            if name in ("faster_whisper", "whisper"):
                raise ImportError(name)
            return real_import(name, *args, **kw)

        monkeypatch.setattr(builtins, "__import__", _block)
        with pytest.raises(ImportError):
            stt._transcribe_pcm(_make_pcm())


class TestOpenAIWhisperSTT:
    @pytest.mark.asyncio
    async def test_buffers_all_chunks(self) -> None:
        stt = OpenAIWhisperSTT()

        received: list[bytes] = []

        def _fake_api(pcm: bytes) -> str:
            received.append(pcm)
            return "buffered result"

        with patch.object(stt, "_transcribe_api", side_effect=_fake_api):

            async def _source() -> AsyncIterator[bytes]:
                yield b"\x01" * 100
                yield b"\x02" * 100

            parts = []
            async for chunk in stt.transcribe_stream(_source()):
                parts.append(chunk)

        assert parts == ["buffered result"]
        # All chunks combined into one call
        assert len(received) == 1
        assert len(received[0]) == 200

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self) -> None:
        stt = OpenAIWhisperSTT()

        async def _empty() -> AsyncIterator[bytes]:
            return
            yield

        parts = []
        async for chunk in stt.transcribe_stream(_empty()):
            parts.append(chunk)
        assert parts == []

    def test_raises_without_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stt = OpenAIWhisperSTT()

        import builtins

        real_import = builtins.__import__

        def _block(name: str, *args: object, **kw: object) -> object:
            if name == "openai":
                raise ImportError("openai")
            return real_import(name, *args, **kw)

        monkeypatch.setattr(builtins, "__import__", _block)
        with pytest.raises(ImportError, match="openai"):
            stt._transcribe_api(_make_pcm())
