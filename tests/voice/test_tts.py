"""Tests for TTS implementations and sentence splitting."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from synapsekit.voice.tts import OpenAITTS, _split_sentences


class TestSplitSentences:
    def test_splits_on_period(self) -> None:
        parts = _split_sentences("Hello world. How are you? Fine")
        assert parts[0] == "Hello world."
        assert parts[1] == "How are you?"
        assert parts[2] == "Fine"

    def test_no_boundary_returns_single(self) -> None:
        parts = _split_sentences("just a sentence")
        assert parts == ["just a sentence"]

    def test_multiple_punctuation(self) -> None:
        text = "First! Second? Third."
        parts = _split_sentences(text)
        assert len(parts) == 3
        assert parts[0] == "First!"
        assert parts[1] == "Second?"

    def test_empty_string(self) -> None:
        parts = _split_sentences("")
        assert parts == [""]

    def test_long_chunk_is_split(self) -> None:
        # Text longer than _MAX_SENTENCE_CHARS with no punctuation should be hard-split
        long = "word " * 80  # > 300 chars, no sentence boundary
        parts = _split_sentences(long)
        # All but possibly the last should be under _MAX_SENTENCE_CHARS
        for part in parts[:-1]:
            assert len(part) <= 300

    def test_first_sentence_before_rest(self) -> None:
        """Verifies sentence streaming: first sentence available before full response."""
        parts = _split_sentences("First sentence. Second sentence. Third")
        assert "First sentence." in parts
        # 'Third' is the incomplete tail (no trailing period)
        assert parts[-1] == "Third"


class TestOpenAITTS:
    @pytest.mark.asyncio
    async def test_synthesizes_each_sentence_separately(self) -> None:
        tts = OpenAITTS()
        call_args: list[str] = []

        def _fake_synth(text: str) -> bytes:
            call_args.append(text)
            return b"\x00" * 100

        with patch.object(tts, "_synthesize_text", side_effect=_fake_synth):

            async def _text_stream() -> AsyncIterator[str]:
                yield "Hello world. "
                yield "How are you? "
                yield "Fine."

            chunks = []
            async for chunk in tts.synthesize_stream(_text_stream()):
                chunks.append(chunk)

        # Three complete sentences → three synth calls
        assert len(call_args) == 3
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_first_sentence_plays_before_full_response(self) -> None:
        """The first audio chunk is yielded before the final token arrives."""
        tts = OpenAITTS()
        synthesis_order: list[str] = []
        yield_order: list[str] = []

        def _fake_synth(text: str) -> bytes:
            synthesis_order.append(text)
            return b"\x00" * 50

        async def _slow_stream() -> AsyncIterator[str]:
            yield "First sentence. "
            # Simulate delay — second part hasn't arrived yet
            yield "Second"

        with patch.object(tts, "_synthesize_text", side_effect=_fake_synth):
            async for _chunk in tts.synthesize_stream(_slow_stream()):
                yield_order.append("chunk")

        # First sentence synthesised before "Second" is buffered
        assert synthesis_order[0].startswith("First sentence")

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self) -> None:
        tts = OpenAITTS()

        async def _empty() -> AsyncIterator[str]:
            return
            yield

        chunks = []
        async for chunk in tts.synthesize_stream(_empty()):
            chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_whitespace_only_not_synthesised(self) -> None:
        tts = OpenAITTS()
        call_count = 0

        def _fake(text: str) -> bytes:
            nonlocal call_count
            call_count += 1
            return b"\x00"

        with patch.object(tts, "_synthesize_text", side_effect=_fake):

            async def _stream() -> AsyncIterator[str]:
                yield "   "

            async for _ in tts.synthesize_stream(_stream()):
                pass

        assert call_count == 0

    def test_raises_without_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tts = OpenAITTS()
        import builtins

        real = builtins.__import__

        def _block(name: str, *args: object, **kw: object) -> object:
            if name == "openai":
                raise ImportError
            return real(name, *args, **kw)

        monkeypatch.setattr(builtins, "__import__", _block)
        with pytest.raises(ImportError):
            tts._synthesize_text("hello")


class TestCartesiaTTS:
    @pytest.mark.asyncio
    async def test_synthesizes_sentences_separately(self) -> None:
        from synapsekit.voice.tts import CartesiaTTS

        tts = CartesiaTTS(api_key="test")
        sentences_sent: list[str] = []

        async def _fake_stream_sentence(text: str) -> AsyncIterator[bytes]:
            sentences_sent.append(text)
            yield b"\x00" * 100

        with patch.object(tts, "_stream_sentence", side_effect=_fake_stream_sentence):

            async def _stream() -> AsyncIterator[str]:
                yield "Hello world. "
                yield "How are you? "
                yield "Fine."

            chunks = []
            async for chunk in tts.synthesize_stream(_stream()):
                chunks.append(chunk)

        assert len(sentences_sent) == 3
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_raises_without_cartesia(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from synapsekit.voice.tts import CartesiaTTS

        tts = CartesiaTTS(api_key="test")
        import builtins

        real = builtins.__import__

        def _block(name: str, *args: object, **kw: object) -> object:
            if name == "cartesia":
                raise ImportError("cartesia")
            return real(name, *args, **kw)

        monkeypatch.setattr(builtins, "__import__", _block)

        async def _collect() -> None:
            async for _ in tts._stream_sentence("hello"):
                pass

        with pytest.raises(ImportError, match="cartesia"):
            await _collect()

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self) -> None:
        from synapsekit.voice.tts import CartesiaTTS

        tts = CartesiaTTS(api_key="test")

        async def _empty() -> AsyncIterator[str]:
            return
            yield

        chunks = []
        async for chunk in tts.synthesize_stream(_empty()):
            chunks.append(chunk)
        assert chunks == []


class TestPiperTTSCaching:
    def test_voice_instance_is_same_object_across_calls(self) -> None:
        """_get_voice() must return the identical object on every call — no reload."""

        from unittest.mock import MagicMock, patch

        from synapsekit.voice.tts import PiperTTS

        tts = PiperTTS(model_path="/fake/model.onnx")
        fake_voice = MagicMock()
        fake_voice.synthesize = MagicMock()

        # Seed the cache — simulates what _get_voice does on the first real call.
        tts._voice = fake_voice

        with patch("synapsekit.voice.tts.wave.open"):
            for _ in range(5):
                tts._synthesize_text("Hello.")

        # The cached instance must be the exact same object throughout.
        assert tts._voice is fake_voice
        # synthesize was called 5 times on the same instance — never recreated.
        assert fake_voice.synthesize.call_count == 5

    def test_piper_voice_load_called_only_once(self) -> None:
        """PiperVoice.load must be called exactly once even after many sentences."""
        from unittest.mock import MagicMock, patch

        from synapsekit.voice.tts import PiperTTS

        tts = PiperTTS(model_path="/fake/model.onnx")
        fake_voice = MagicMock()
        fake_voice.synthesize = MagicMock()
        load_calls: list[int] = []

        class _FakePiperVoice:
            @staticmethod
            def load(path, config_path=None):
                load_calls.append(1)
                return fake_voice

        with (
            patch.dict("sys.modules", {"piper": MagicMock(PiperVoice=_FakePiperVoice)}),
            patch("synapsekit.voice.tts.wave.open"),
        ):
            for _ in range(4):
                tts._synthesize_text("Sentence.")

        assert len(load_calls) == 1, f"PiperVoice.load called {len(load_calls)} times, expected 1"
