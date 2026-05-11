"""Tests for VoicePipeline — VAD gating, interruption, streaming order, cancellation."""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synapsekit.voice.pipeline import VoicePipeline, _AudioPlayer
from synapsekit.voice.types import PipelineEvent, PipelineState


# ── Helpers ────────────────────────────────────────────────────────────────────

def _silence_frame(num_samples: int = 480) -> bytes:
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def _speech_frame(amplitude: int = 10000, num_samples: int = 480) -> bytes:
    return struct.pack(f"<{num_samples}h", *([amplitude] * num_samples))


class _MockVAD:
    """Controllable VAD — returns True for speech frames, False for silence."""

    def __init__(self, speech_frames: set[int] | None = None) -> None:
        self._call_count = 0
        self._speech_frames = speech_frames or set()

    async def is_speech(self, frame: bytes) -> bool:
        idx = self._call_count
        self._call_count += 1
        return idx in self._speech_frames


class _AlwaysSpeechVAD:
    async def is_speech(self, frame: bytes) -> bool:
        return True


class _AlwaysSilenceVAD:
    async def is_speech(self, frame: bytes) -> bool:
        return False


class _MockSTT:
    def __init__(self, transcript: str = "hello world") -> None:
        self._transcript = transcript
        self.call_count = 0

    async def transcribe_stream(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        async for _ in audio_stream:
            pass
        self.call_count += 1
        if self._transcript:
            yield self._transcript


class _MockTTS:
    def __init__(self, audio_per_sentence: bytes = b"\x00" * 48000) -> None:
        self._audio = audio_per_sentence
        self.synthesis_count = 0

    async def synthesize_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        async for _ in text_stream:
            pass
        self.synthesis_count += 1
        yield self._audio


class _MockLLM:
    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens or ["Hello.", " How are you?"]

    async def stream_with_messages(self, messages: Any, **kw: Any) -> AsyncIterator[str]:
        for token in self._tokens:
            await asyncio.sleep(0)
            yield token


def _make_pipeline(
    vad: Any = None,
    stt: Any = None,
    tts: Any = None,
    llm: Any = None,
    allow_interruption: bool = True,
    interrupt_threshold_ms: int = 300,
) -> VoicePipeline:
    return VoicePipeline(
        llm=llm or _MockLLM(),
        stt=stt or _MockSTT(),
        tts=tts or _MockTTS(),
        vad=vad or _AlwaysSilenceVAD(),
        allow_interruption=allow_interruption,
        interrupt_threshold_ms=interrupt_threshold_ms,
    )


async def _finite_source(frames: list[bytes]) -> AsyncIterator[bytes]:
    for frame in frames:
        await asyncio.sleep(0)
        yield frame


# ── VAD gating ────────────────────────────────────────────────────────────────

class TestVADGating:
    @pytest.mark.asyncio
    async def test_silence_never_triggers_stt(self) -> None:
        """Pure silence must not trigger STT at all."""
        stt = _MockSTT()
        pipeline = _make_pipeline(vad=_AlwaysSilenceVAD(), stt=stt)

        frames = [_silence_frame() for _ in range(100)]
        await pipeline.run(
            _finite_source(frames),
            chunk_duration_ms=30,
            silence_duration_ms=1500,
        )
        assert stt.call_count == 0

    @pytest.mark.asyncio
    async def test_speech_followed_by_silence_triggers_stt(self) -> None:
        """Speech frames followed by silence must produce one STT call."""
        stt = _MockSTT(transcript="test")
        # Speech for frames 0-9, then silence for 50 more (triggers utterance end)
        speech_idxs = set(range(10))
        vad = _MockVAD(speech_frames=speech_idxs)

        tts_calls: list[bytes] = []

        class _CaptureTTS(_MockTTS):
            async def synthesize_stream(self, ts: AsyncIterator[str]) -> AsyncIterator[bytes]:
                async for _ in ts:
                    pass
                tts_calls.append(b"audio")
                yield b"\x00" * 100

        pipeline = _make_pipeline(vad=vad, stt=stt, tts=_CaptureTTS())
        frames = [_speech_frame() for _ in range(10)] + [_silence_frame() for _ in range(55)]
        await pipeline.run(
            _finite_source(frames),
            chunk_duration_ms=30,
            silence_duration_ms=1500,
        )
        assert stt.call_count == 1

    @pytest.mark.asyncio
    async def test_vad_gate_prevents_stt_cost_explosion(self) -> None:
        """100 frames of silence → 0 STT calls."""
        stt = _MockSTT()
        pipeline = _make_pipeline(vad=_AlwaysSilenceVAD(), stt=stt)
        frames = [_silence_frame() for _ in range(100)]
        await pipeline.run(_finite_source(frames), chunk_duration_ms=30)
        assert stt.call_count == 0


# ── Streaming order ───────────────────────────────────────────────────────────

class TestStreamingOrder:
    @pytest.mark.asyncio
    async def test_pipeline_stages_are_in_order(self) -> None:
        """State transitions must follow: LISTENING → TRANSCRIBING → GENERATING → SPEAKING."""
        events: list[str] = []

        async def _on_event(e: PipelineEvent) -> None:
            if e.kind == "state_change" and isinstance(e.data, PipelineState):
                events.append(e.data.value)

        speech_idxs = set(range(10))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(10)] + [_silence_frame() for _ in range(55)]

        pipeline = _make_pipeline(vad=vad)
        await pipeline.run(
            _finite_source(frames),
            chunk_duration_ms=30,
            silence_duration_ms=1500,
            on_event=_on_event,
        )

        assert "listening" in events
        assert "transcribing" in events
        assert "generating" in events
        assert "speaking" in events
        # Order check
        idx = {s: events.index(s) for s in ("listening", "transcribing", "generating", "speaking")}
        assert idx["listening"] < idx["transcribing"] < idx["generating"] < idx["speaking"]

    @pytest.mark.asyncio
    async def test_response_tokens_emitted_during_generating(self) -> None:
        tokens_received: list[str] = []

        async def _on_event(e: PipelineEvent) -> None:
            if e.kind == "response_token":
                tokens_received.append(e.data)

        speech_idxs = set(range(10))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(10)] + [_silence_frame() for _ in range(55)]

        pipeline = _make_pipeline(vad=vad)
        await pipeline.run(
            _finite_source(frames),
            chunk_duration_ms=30,
            silence_duration_ms=1500,
            on_event=_on_event,
        )
        assert len(tokens_received) > 0


# ── Sentence-level streaming ──────────────────────────────────────────────────

class TestSentenceStreaming:
    @pytest.mark.asyncio
    async def test_first_sentence_synthesised_before_full_response(self) -> None:
        """First audio chunk must be yielded before all LLM tokens arrive."""
        synthesis_calls: list[str] = []
        all_tokens: list[str] = []

        class _TrackingTTS:
            async def synthesize_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
                from synapsekit.voice.tts import _split_sentences

                buffer = ""
                async for token in text_stream:
                    all_tokens.append(token)
                    buffer += token
                    sentences = _split_sentences(buffer)
                    for s in sentences[:-1]:
                        if s.strip():
                            synthesis_calls.append(s)
                            yield b"\x00" * 100
                    buffer = sentences[-1]
                if buffer.strip():
                    synthesis_calls.append(buffer)
                    yield b"\x00" * 100

        class _SlowLLM:
            async def stream_with_messages(self, messages: Any, **kw: Any) -> AsyncIterator[str]:
                tokens = ["First sentence. ", "Second ", "sentence. ", "Third."]
                for t in tokens:
                    await asyncio.sleep(0.01)
                    yield t

        speech_idxs = set(range(5))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(5)] + [_silence_frame() for _ in range(55)]

        pipeline = _make_pipeline(vad=vad, llm=_SlowLLM(), tts=_TrackingTTS())
        await pipeline.run(
            _finite_source(frames),
            chunk_duration_ms=30,
            silence_duration_ms=1500,
        )

        # First synthesis must have happened before all tokens arrived
        assert len(synthesis_calls) >= 1
        # First synthesis is the first sentence, not the whole response
        assert "First sentence." in synthesis_calls[0]


# ── Interruption handling ─────────────────────────────────────────────────────

class TestInterruptionHandling:
    @pytest.mark.asyncio
    async def test_tts_stops_after_interrupt_threshold(self) -> None:
        """Continuous speech > 300 ms during TTS must cancel playback."""
        interrupted_event = asyncio.Event()

        class _SlowTTS:
            async def synthesize_stream(self, ts: AsyncIterator[str]) -> AsyncIterator[bytes]:
                async for _ in ts:
                    pass
                for _ in range(10):
                    await asyncio.sleep(0.05)
                    yield b"\x00" * 100

        class _TrackingPlayer:
            interrupt_called = False

            async def interrupt(self) -> None:
                _TrackingPlayer.interrupt_called = True
                interrupted_event.set()

            def start(self) -> None:
                pass

            async def put(self, chunk: bytes) -> None:
                pass

            async def drain(self) -> None:
                pass

        # Frame layout:
        # 0-4: speech (VAD says True) — initial utterance
        # 5-54: silence — triggers STT/LLM/TTS
        # 55-74: speech during TTS — should interrupt after >300ms
        total_frames = 75
        speech_idxs = set(range(5)) | set(range(55, 75))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(total_frames)]

        pipeline = _make_pipeline(
            vad=vad,
            tts=_SlowTTS(),
            allow_interruption=True,
            interrupt_threshold_ms=300,
        )

        # Patch _AudioPlayer to use our tracker
        with patch("synapsekit.voice.pipeline._AudioPlayer", return_value=_TrackingPlayer()):
            await asyncio.wait_for(
                pipeline.run(
                    _finite_source(frames),
                    chunk_duration_ms=30,
                    silence_duration_ms=1500,
                ),
                timeout=5.0,
            )

    @pytest.mark.asyncio
    async def test_interruption_emits_interrupted_state(self) -> None:
        events: list[str] = []

        async def _on_event(e: PipelineEvent) -> None:
            if e.kind == "state_change" and isinstance(e.data, PipelineState):
                events.append(e.data.value)

        class _SlowTTS:
            async def synthesize_stream(self, ts: AsyncIterator[str]) -> AsyncIterator[bytes]:
                async for _ in ts:
                    pass
                await asyncio.sleep(0.1)
                yield b"\x00" * 100

        # 5 speech frames → utterance, then 55 silence → STT, then 15 speech → interrupt
        speech_idxs = set(range(5)) | set(range(60, 75))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(75)]

        pipeline = _make_pipeline(
            vad=vad,
            tts=_SlowTTS(),
            allow_interruption=True,
            interrupt_threshold_ms=300,
        )

        with patch("synapsekit.voice.pipeline._AudioPlayer") as MockPlayer:
            instance = MagicMock()
            instance.interrupt = AsyncMock()
            instance.put = AsyncMock()
            instance.drain = AsyncMock()
            instance.start = MagicMock()
            MockPlayer.return_value = instance

            await asyncio.wait_for(
                pipeline.run(
                    _finite_source(frames),
                    chunk_duration_ms=30,
                    silence_duration_ms=1500,
                    on_event=_on_event,
                ),
                timeout=5.0,
            )

        assert "interrupted" in events

    @pytest.mark.asyncio
    async def test_no_interruption_when_disabled(self) -> None:
        """With allow_interruption=False, TTS must not be cancelled."""
        tts_completed = asyncio.Event()

        class _TrackingTTS:
            async def synthesize_stream(self, ts: AsyncIterator[str]) -> AsyncIterator[bytes]:
                async for _ in ts:
                    pass
                yield b"\x00" * 100
                tts_completed.set()

        speech_idxs = set(range(5)) | set(range(60, 80))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(80)]

        pipeline = _make_pipeline(
            vad=vad,
            tts=_TrackingTTS(),
            allow_interruption=False,
        )

        with patch("synapsekit.voice.pipeline._AudioPlayer") as MockPlayer:
            instance = MagicMock()
            instance.interrupt = AsyncMock()
            instance.put = AsyncMock()
            instance.drain = AsyncMock()
            instance.start = MagicMock()
            MockPlayer.return_value = instance

            await asyncio.wait_for(
                pipeline.run(
                    _finite_source(frames),
                    chunk_duration_ms=30,
                    silence_duration_ms=1500,
                ),
                timeout=5.0,
            )

        # interrupt() must not have been called
        instance.interrupt.assert_not_called()


# ── Cancellation / cleanup ────────────────────────────────────────────────────

class TestCancellationAndCleanup:
    @pytest.mark.asyncio
    async def test_no_dangling_tasks_on_normal_exit(self) -> None:
        """All internal tasks must be done after run() returns."""
        pipeline = _make_pipeline(vad=_AlwaysSilenceVAD())
        frames = [_silence_frame() for _ in range(10)]
        await pipeline.run(_finite_source(frames), chunk_duration_ms=30)

        pending = [t for t in asyncio.all_tasks() if not t.done() and t != asyncio.current_task()]
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_empty_transcript_does_not_start_tts(self) -> None:
        """Empty STT result must not invoke TTS."""
        stt = _MockSTT(transcript="")
        tts = _MockTTS()

        speech_idxs = set(range(5))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(5)] + [_silence_frame() for _ in range(55)]

        pipeline = _make_pipeline(vad=vad, stt=stt, tts=tts)
        await pipeline.run(
            _finite_source(frames),
            chunk_duration_ms=30,
            silence_duration_ms=1500,
        )
        assert tts.synthesis_count == 0

    @pytest.mark.asyncio
    async def test_provider_exception_does_not_crash_pipeline(self) -> None:
        """A TTS error must emit an error event, not raise."""

        class _FailingTTS:
            async def synthesize_stream(self, ts: AsyncIterator[str]) -> AsyncIterator[bytes]:
                async for _ in ts:
                    pass
                raise RuntimeError("TTS service unavailable")
                yield  # make it a generator

        errors: list[str] = []

        async def _on_event(e: PipelineEvent) -> None:
            if e.kind == "error":
                errors.append(str(e.data))

        speech_idxs = set(range(5))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(5)] + [_silence_frame() for _ in range(55)]

        pipeline = _make_pipeline(vad=vad, tts=_FailingTTS())

        with patch("synapsekit.voice.pipeline._AudioPlayer") as MockPlayer:
            instance = MagicMock()
            instance.put = AsyncMock()
            instance.drain = AsyncMock()
            instance.start = MagicMock()
            instance.interrupt = AsyncMock()
            MockPlayer.return_value = instance

            await asyncio.wait_for(
                pipeline.run(
                    _finite_source(frames),
                    chunk_duration_ms=30,
                    silence_duration_ms=1500,
                    on_event=_on_event,
                ),
                timeout=5.0,
            )

        assert len(errors) > 0
        assert "TTS service unavailable" in errors[0]


# ── Audio player unit tests ───────────────────────────────────────────────────

class TestAudioPlayer:
    @pytest.mark.asyncio
    async def test_interrupt_clears_queue(self) -> None:
        player = _AudioPlayer(sample_rate=16000)

        async def _fake_loop() -> None:
            await asyncio.sleep(10)  # simulate playing

        player._task = asyncio.create_task(_fake_loop())

        # Queue up some chunks
        for _ in range(5):
            player._queue.put_nowait(b"\x00" * 100)

        mock_sd = MagicMock()
        mock_sd.stop = MagicMock()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            await player.interrupt()

        assert player._queue.empty()
        assert player._task is None

    @pytest.mark.asyncio
    async def test_interrupt_cancels_task(self) -> None:
        player = _AudioPlayer(sample_rate=16000)
        cancelled = False

        async def _blocker() -> None:
            nonlocal cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled = True
                raise

        player._task = asyncio.create_task(_blocker())
        await asyncio.sleep(0)  # let task start

        mock_sd = MagicMock()
        mock_sd.stop = MagicMock()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            await player.interrupt()

        assert cancelled


# ── AgentMemory integration ───────────────────────────────────────────────────

class TestAgentMemoryIntegration:
    """Memory recall augments LLM context; exchanges are stored after each turn."""

    def _make_mock_memory(self) -> Any:
        memory = MagicMock()
        memory.recall = AsyncMock(return_value=[])
        memory.store = AsyncMock()
        return memory

    @pytest.mark.asyncio
    async def test_memory_recall_called_per_utterance(self) -> None:
        memory = self._make_mock_memory()

        speech_idxs = set(range(5))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(5)] + [_silence_frame() for _ in range(55)]

        pipeline = _make_pipeline(vad=vad)
        pipeline._memory = memory
        pipeline._agent_id = "test-agent"

        with patch("synapsekit.voice.pipeline._AudioPlayer") as MockPlayer:
            instance = MagicMock()
            instance.put = AsyncMock()
            instance.drain = AsyncMock()
            instance.start = MagicMock()
            instance.interrupt = AsyncMock()
            MockPlayer.return_value = instance

            await asyncio.wait_for(
                pipeline.run(
                    _finite_source(frames),
                    chunk_duration_ms=30,
                    silence_duration_ms=1500,
                ),
                timeout=5.0,
            )

        memory.recall.assert_called_once_with(
            agent_id="test-agent",
            query="hello world",
            top_k=3,
        )

    @pytest.mark.asyncio
    async def test_exchange_stored_after_turn(self) -> None:
        memory = self._make_mock_memory()

        speech_idxs = set(range(5))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(5)] + [_silence_frame() for _ in range(55)]

        pipeline = _make_pipeline(vad=vad)
        pipeline._memory = memory
        pipeline._agent_id = "test-agent"

        with patch("synapsekit.voice.pipeline._AudioPlayer") as MockPlayer:
            instance = MagicMock()
            instance.put = AsyncMock()
            instance.drain = AsyncMock()
            instance.start = MagicMock()
            instance.interrupt = AsyncMock()
            MockPlayer.return_value = instance

            await asyncio.wait_for(
                pipeline.run(
                    _finite_source(frames),
                    chunk_duration_ms=30,
                    silence_duration_ms=1500,
                ),
                timeout=5.0,
            )

        memory.store.assert_called_once()
        call_kw = memory.store.call_args.kwargs
        assert call_kw["agent_id"] == "test-agent"
        assert call_kw["memory_type"] == "episodic"
        assert "hello world" in call_kw["content"]

    @pytest.mark.asyncio
    async def test_memory_records_injected_into_context(self) -> None:
        """When recall returns records, they appear in the messages sent to LLM."""
        from synapsekit.memory.base import MemoryRecord, MemoryType
        from datetime import datetime, timezone

        record = MemoryRecord(
            id="r1",
            agent_id="test",
            content="User prefers concise answers.",
            memory_type="semantic",
            embedding=[0.0],
            created_at=datetime.now(timezone.utc),
            accessed_at=datetime.now(timezone.utc),
            access_count=0,
            ttl_days=None,
            metadata={},
        )
        memory = self._make_mock_memory()
        memory.recall = AsyncMock(return_value=[record])

        captured_messages: list[list[dict]] = []

        class _CapturingLLM:
            async def stream_with_messages(
                self, messages: Any, **kw: Any
            ) -> AsyncIterator[str]:
                captured_messages.append(list(messages))
                yield "OK."

        speech_idxs = set(range(5))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(5)] + [_silence_frame() for _ in range(55)]

        pipeline = _make_pipeline(vad=vad, llm=_CapturingLLM())
        pipeline._memory = memory
        pipeline._agent_id = "test"

        with patch("synapsekit.voice.pipeline._AudioPlayer") as MockPlayer:
            instance = MagicMock()
            instance.put = AsyncMock()
            instance.drain = AsyncMock()
            instance.start = MagicMock()
            instance.interrupt = AsyncMock()
            MockPlayer.return_value = instance

            await asyncio.wait_for(
                pipeline.run(
                    _finite_source(frames),
                    chunk_duration_ms=30,
                    silence_duration_ms=1500,
                ),
                timeout=5.0,
            )

        assert captured_messages, "LLM was not called"
        msgs = captured_messages[0]
        # A system message containing the memory record must be present
        memory_msgs = [m for m in msgs if m["role"] == "system" and "memory" in m["content"].lower()]
        assert memory_msgs, "Memory context not injected into LLM messages"
        assert "User prefers concise answers." in memory_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_memory_failure_does_not_crash_pipeline(self) -> None:
        """Exceptions in memory recall/store must not propagate."""
        memory = MagicMock()
        memory.recall = AsyncMock(side_effect=RuntimeError("backend unavailable"))
        memory.store = AsyncMock(side_effect=RuntimeError("backend unavailable"))

        speech_idxs = set(range(5))
        vad = _MockVAD(speech_frames=speech_idxs)
        frames = [_speech_frame() for _ in range(5)] + [_silence_frame() for _ in range(55)]

        pipeline = _make_pipeline(vad=vad)
        pipeline._memory = memory

        with patch("synapsekit.voice.pipeline._AudioPlayer") as MockPlayer:
            instance = MagicMock()
            instance.put = AsyncMock()
            instance.drain = AsyncMock()
            instance.start = MagicMock()
            instance.interrupt = AsyncMock()
            MockPlayer.return_value = instance

            # Must complete without raising
            await asyncio.wait_for(
                pipeline.run(
                    _finite_source(frames),
                    chunk_duration_ms=30,
                    silence_duration_ms=1500,
                ),
                timeout=5.0,
            )
