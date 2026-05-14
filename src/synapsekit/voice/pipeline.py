"""
Real-time VoicePipeline.

Pipeline flow::

    microphone → VAD → STT → LLM → TTS → speaker

VAD gates transcription so silence is never forwarded to STT providers,
keeping cloud costs low and latency sharp.

Interruption handling stops TTS playback within ``interrupt_threshold_ms``
of continuous speech onset, then opens a fresh STT session for the new
utterance.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..memory.agent_memory import AgentMemory

from ..llm.base import BaseLLM
from .base import BaseSTT, BaseTTS, BaseVAD
from .types import PipelineEvent, PipelineState

# AgentMemory is optional — import lazily to avoid pulling in memory backends
# when the caller does not use persistent sessions.
try:
    from ..memory.agent_memory import AgentMemory as _AgentMemory
except Exception:  # pragma: no cover
    _AgentMemory = None  # type: ignore[assignment,misc]


class _AudioPlayer:
    """
    Non-blocking PCM audio player backed by sounddevice.

    Chunks are queued and played sequentially in a background task.
    ``interrupt()`` stops current playback instantly and discards all
    buffered chunks, ensuring no stale assistant audio leaks after an
    interruption.
    """

    def __init__(self, sample_rate: int = 24000) -> None:
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def put(self, chunk: bytes) -> None:
        await self._queue.put(chunk)

    async def drain(self) -> None:
        """Signal end-of-stream and wait for all queued audio to finish."""
        await self._queue.put(None)
        if self._task and not self._task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def interrupt(self) -> None:
        """Stop immediately and discard all pending audio."""
        # Drain the queue first so the play loop doesn't pick up stale chunks
        while not self._queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()

        # Stop sounddevice hardware playback
        try:
            import sounddevice as sd  # type: ignore[import]

            await asyncio.to_thread(sd.stop)
        except (ImportError, ModuleNotFoundError, Exception):
            pass

        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._task = None

    async def _loop(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice and numpy are required for audio playback. "
                "Install with: pip install 'synapsekit[voice-stream]'"
            ) from None

        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            try:
                samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0

                def _play(s: np.ndarray = samples) -> None:
                    sd.play(s, self._sample_rate)
                    sd.wait()

                await asyncio.to_thread(_play)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[VoicePipeline] playback error: {exc}", file=sys.stderr)


class VoicePipeline:
    """
    Production-grade real-time voice pipeline.

    Composes a VAD, STT, LLM, and TTS provider into a single streaming
    conversation loop::

        microphone → VAD → STT → LLM → TTS → speaker

    Key behaviours
    ~~~~~~~~~~~~~~
    * **VAD gating** — silence is never forwarded to STT; STT sessions open
      only when VAD detects speech onset.
    * **Sentence-level TTS streaming** — LLM output is parsed at sentence
      boundaries.  The first sentence plays before the full response is
      generated, keeping perceived latency under 500 ms.
    * **Interruption handling** — if the user speaks for longer than
      ``interrupt_threshold_ms`` while the assistant is speaking, TTS is
      cancelled immediately, audio buffers are flushed, and a new STT session
      opens.  The default 300 ms debounce prevents clicks and transient
      sounds from triggering false interruptions.
    * **Conversational memory** — per-turn history is kept in-process.
      Pass an :class:`~synapsekit.memory.agent_memory.AgentMemory` instance
      for cross-session persistence: relevant memories are recalled before
      each LLM call and each exchange is stored as an episodic memory.

    Parameters
    ----------
    llm:
        Any :class:`~synapsekit.llm.base.BaseLLM` instance.  Streaming is
        used via ``stream_with_messages``; the provider must support it.
    stt:
        :class:`~synapsekit.voice.base.BaseSTT` implementation.
    tts:
        :class:`~synapsekit.voice.base.BaseTTS` implementation.
    vad:
        :class:`~synapsekit.voice.base.BaseVAD` implementation.
    allow_interruption:
        Enable mid-speech interruption detection.  Default ``True``.
    interrupt_threshold_ms:
        Continuous speech duration (ms) required to trigger an interruption.
        Default 300 ms — prevents clicks and keyboard sounds from interrupting
        while still feeling responsive in conversation.
    memory:
        Optional :class:`~synapsekit.memory.agent_memory.AgentMemory` for
        persistent cross-session memory.  When provided, relevant past
        exchanges are retrieved before each LLM call, and the current
        exchange is stored as an episodic memory after the assistant responds.
    agent_id:
        Identifier used to namespace memories in the backend.  Different
        voice sessions can share a backend by using distinct IDs.
    memory_top_k:
        Number of memory records to retrieve per turn.  Default 3.
    """

    def __init__(
        self,
        llm: BaseLLM,
        stt: BaseSTT,
        tts: BaseTTS,
        vad: BaseVAD,
        allow_interruption: bool = True,
        interrupt_threshold_ms: int = 300,
        memory: AgentMemory | None = None,
        agent_id: str = "voice_agent",
        memory_top_k: int = 3,
    ) -> None:
        self._llm = llm
        self._stt = stt
        self._tts = tts
        self._vad = vad
        self._allow_interruption = allow_interruption
        self._interrupt_threshold_ms = interrupt_threshold_ms
        self._memory = memory
        self._agent_id = agent_id
        self._memory_top_k = memory_top_k
        self._messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_system_prompt(self, prompt: str) -> None:
        """Set (or replace) the system message at the start of history."""
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = prompt
        else:
            self._messages.insert(0, {"role": "system", "content": prompt})

    def reset_history(self) -> None:
        """Clear all conversation history except a system prompt if present."""
        system = [m for m in self._messages if m.get("role") == "system"]
        self._messages = system

    async def run(
        self,
        audio_source: AsyncIterator[bytes],
        *,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 30,
        silence_duration_ms: int = 1500,
        tts_sample_rate: int = 24000,
        system_prompt: str | None = None,
        on_event: Callable[[PipelineEvent], Awaitable[None]] | None = None,
    ) -> None:
        """
        Run the pipeline until *audio_source* is exhausted.

        Parameters
        ----------
        audio_source:
            Async iterator of raw 16-bit mono PCM frames captured at
            *sample_rate* Hz.  Each frame should be approximately
            *chunk_duration_ms* ms of audio.
        sample_rate:
            Sample rate of incoming audio in Hz.  Default 16000.
        chunk_duration_ms:
            Duration of each audio frame in ms.  Used to compute silence
            detection windows and interruption debounce timing.  Default 30.
        silence_duration_ms:
            Continuous silence after speech required to finalise an utterance
            and forward it to STT.  Default 1500 ms.
        tts_sample_rate:
            Sample rate of audio emitted by the TTS provider.  Default 24000
            (OpenAI TTS PCM output).  Adjust when using PiperTTS (22050) or
            ElevenLabs PCM modes.
        system_prompt:
            If provided, replaces the current system message in history.
        on_event:
            Optional async callback invoked for each :class:`PipelineEvent`.
            Useful for logging, UI updates, or telemetry.
        """
        if system_prompt:
            self.set_system_prompt(system_prompt)

        silence_chunks_needed = max(1, silence_duration_ms // chunk_duration_ms)

        # Bounded queue buffers mic frames so reads and processing run concurrently
        frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=2000)

        async def _reader() -> None:
            try:
                async for frame in audio_source:
                    await frame_queue.put(frame)
            finally:
                await frame_queue.put(None)

        reader_task = asyncio.create_task(_reader())
        try:
            await self._pipeline_loop(
                frame_queue,
                sample_rate=sample_rate,
                chunk_duration_ms=chunk_duration_ms,
                silence_chunks_needed=silence_chunks_needed,
                tts_sample_rate=tts_sample_rate,
                on_event=on_event,
            )
        except KeyboardInterrupt:
            pass
        finally:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader_task

    async def run_microphone(
        self,
        *,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 30,
        silence_duration_ms: int = 1500,
        tts_sample_rate: int = 24000,
        system_prompt: str | None = None,
        on_event: Callable[[PipelineEvent], Awaitable[None]] | None = None,
    ) -> None:
        """
        Convenience wrapper: open the system microphone and run the pipeline.

        Requires ``sounddevice``.  Install with::

            pip install 'synapsekit[voice-stream]'
        """
        try:
            import sounddevice as sd  # noqa: F401
        except ImportError:
            raise ImportError(
                "sounddevice is required for run_microphone. "
                "Install with: pip install 'synapsekit[voice-stream]'"
            ) from None

        import queue as _queue

        chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        q: _queue.Queue[bytes] = _queue.Queue()

        def _cb(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))

        async def _mic_source() -> AsyncIterator[bytes]:
            import sounddevice as sd

            with sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=chunk_size,
                dtype="int16",
                channels=1,
                callback=_cb,
            ):
                while True:
                    try:
                        frame = await asyncio.to_thread(q.get, True, 0.02)
                        yield frame
                    except _queue.Empty:
                        pass

        await self.run(
            _mic_source(),
            sample_rate=sample_rate,
            chunk_duration_ms=chunk_duration_ms,
            silence_duration_ms=silence_duration_ms,
            tts_sample_rate=tts_sample_rate,
            system_prompt=system_prompt,
            on_event=on_event,
        )

    # ------------------------------------------------------------------ #
    # Memory helpers                                                       #
    # ------------------------------------------------------------------ #

    async def _build_messages_with_memory(self, query: str) -> list[dict[str, Any]]:
        """
        Return a copy of ``_messages`` with recalled memory records injected
        as an additional system context block immediately after the primary
        system message (if any).

        Memory failures are silently suppressed so they never crash the
        pipeline — the call falls back to the unaugmented history.
        """
        messages = list(self._messages)
        if self._memory is None:
            return messages
        try:
            records = await self._memory.recall(
                agent_id=self._agent_id,
                query=query,
                top_k=self._memory_top_k,
            )
        except Exception:
            return messages

        if not records:
            return messages

        block = "\n".join(f"- {r.content}" for r in records)
        context = {
            "role": "system",
            "content": f"Relevant context recalled from memory:\n{block}",
        }
        if messages and messages[0].get("role") == "system":
            return [messages[0], context, *messages[1:]]
        return [context, *messages]

    async def _store_exchange(self, transcript: str, response: str) -> None:
        """Persist a user/assistant exchange as an episodic memory record."""
        if self._memory is None or not transcript or not response:
            return
        with contextlib.suppress(Exception):
            await self._memory.store(
                agent_id=self._agent_id,
                content=f"User: {transcript}\nAssistant: {response}",
                memory_type="episodic",
            )

    # ------------------------------------------------------------------ #
    # Internal pipeline machinery                                          #
    # ------------------------------------------------------------------ #

    async def _pipeline_loop(
        self,
        frame_queue: asyncio.Queue[bytes | None],
        *,
        sample_rate: int,
        chunk_duration_ms: int,
        silence_chunks_needed: int,
        tts_sample_rate: int,
        on_event: Callable[[PipelineEvent], Awaitable[None]] | None,
    ) -> None:
        async def _emit(kind: str, data: Any = None) -> None:
            if on_event:
                await on_event(PipelineEvent(kind, data))

        speech_buffer = bytearray()
        is_listening = False
        silence_count = 0

        # Active TTS session handles
        tts_player: _AudioPlayer | None = None
        llm_task: asyncio.Task[None] | None = None
        tts_task: asyncio.Task[None] | None = None
        interrupt_speech_ms = 0

        while True:
            frame = await frame_queue.get()
            if frame is None:
                break

            # ── Check if TTS finished naturally ──────────────────────────
            if tts_task is not None and tts_task.done():
                tts_player = None
                llm_task = None
                tts_task = None
                interrupt_speech_ms = 0
                await _emit("state_change", PipelineState.IDLE)

            detected = await self._vad.is_speech(frame)

            # ── Interruption monitoring during TTS playback ──────────────
            if tts_player is not None:
                if self._allow_interruption:
                    if detected:
                        interrupt_speech_ms += chunk_duration_ms
                        if interrupt_speech_ms >= self._interrupt_threshold_ms:
                            # Cancel both tasks concurrently to avoid serialised awaits
                            _to_cancel = [t for t in (llm_task, tts_task) if t and not t.done()]
                            for _t in _to_cancel:
                                _t.cancel()
                            if _to_cancel:
                                await asyncio.gather(*_to_cancel, return_exceptions=True)

                            await tts_player.interrupt()
                            tts_player = None
                            llm_task = None
                            tts_task = None
                            interrupt_speech_ms = 0

                            await _emit("state_change", PipelineState.INTERRUPTED)
                            await _emit("state_change", PipelineState.LISTENING)

                            # Seed the new utterance with the frame that triggered interrupt
                            is_listening = True
                            silence_count = 0
                            speech_buffer.clear()
                            speech_buffer.extend(frame)
                    else:
                        interrupt_speech_ms = 0
                # Always skip normal VAD accumulation while TTS is active,
                # regardless of allow_interruption — prevents overlapping TTS.
                continue

            # ── Normal VAD → utterance accumulation ─────────────────────
            if detected:
                if not is_listening:
                    is_listening = True
                    silence_count = 0
                    speech_buffer.clear()
                    await _emit("state_change", PipelineState.LISTENING)

                speech_buffer.extend(frame)
                silence_count = 0

            elif is_listening:
                silence_count += 1
                speech_buffer.extend(frame)

                if silence_count >= silence_chunks_needed:
                    utterance = bytes(speech_buffer)
                    speech_buffer.clear()
                    is_listening = False
                    silence_count = 0

                    result = await self._handle_utterance(
                        utterance,
                        tts_sample_rate=tts_sample_rate,
                        emit=_emit,
                    )
                    if result is not None:
                        tts_player, llm_task, tts_task = result
                        interrupt_speech_ms = 0

        # ── Cleanup on stream end ────────────────────────────────────────
        _cleanup = [t for t in (llm_task, tts_task) if t and not t.done()]
        for _t in _cleanup:
            _t.cancel()
        if _cleanup:
            await asyncio.gather(*_cleanup, return_exceptions=True)
        if tts_player:
            await tts_player.interrupt()

    async def _handle_utterance(
        self,
        utterance: bytes,
        *,
        tts_sample_rate: int,
        emit: Callable[[str, Any], Awaitable[None]],
    ) -> tuple[_AudioPlayer, asyncio.Task[None], asyncio.Task[None]] | None:
        """
        STT → LLM → TTS for one utterance.

        Returns ``(player, llm_task, tts_task)`` so the caller can track and
        cancel both tasks on interruption.  Returns ``None`` if STT produces
        no transcript (silence mis-classified as speech).
        """
        # ── STT ─────────────────────────────────────────────────────────
        await emit("state_change", PipelineState.TRANSCRIBING)

        transcript_parts: list[str] = []

        async def _utterance_source() -> AsyncIterator[bytes]:
            yield utterance

        try:
            async for chunk in self._stt.transcribe_stream(_utterance_source()):
                transcript_parts.append(chunk)
                await emit("transcript", chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await emit("error", f"STT error: {exc}")
            await emit("state_change", PipelineState.IDLE)
            return None

        transcript = "".join(transcript_parts).strip()
        if not transcript:
            await emit("state_change", PipelineState.IDLE)
            return None

        # ── LLM ─────────────────────────────────────────────────────────
        self._messages.append({"role": "user", "content": transcript})
        await emit("state_change", PipelineState.GENERATING)

        # Augment with recalled memories (non-blocking on failure)
        messages_for_llm = await self._build_messages_with_memory(transcript)

        # Token queue bridges LLM streaming → TTS streaming
        token_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=512)

        # Capture transcript in closure for memory storage after response
        _transcript = transcript

        async def _llm_stream() -> None:
            response_tokens: list[str] = []
            try:
                async for token in self._llm.stream_with_messages(messages_for_llm):
                    response_tokens.append(token)
                    await token_queue.put(token)
                    await emit("response_token", token)
            except asyncio.CancelledError:
                pass
            finally:
                await token_queue.put(None)  # always unblock TTS
                if response_tokens:
                    response_text = "".join(response_tokens)
                    self._messages.append({"role": "assistant", "content": response_text})
                    await self._store_exchange(_transcript, response_text)

        async def _token_stream() -> AsyncIterator[str]:
            while True:
                token = await token_queue.get()
                if token is None:
                    break
                yield token

        # ── TTS ─────────────────────────────────────────────────────────
        player = _AudioPlayer(sample_rate=tts_sample_rate)
        player.start()

        async def _tts_gen() -> None:
            await emit("state_change", PipelineState.SPEAKING)
            try:
                async for audio_chunk in self._tts.synthesize_stream(_token_stream()):
                    await player.put(audio_chunk)
                    await emit("audio_chunk", len(audio_chunk))
                await player.drain()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await emit("error", str(exc))
                await player.interrupt()  # prevent player task from hanging on TTS error
            finally:
                await emit("state_change", PipelineState.IDLE)

        llm_task = asyncio.create_task(_llm_stream())
        tts_task = asyncio.create_task(_tts_gen())
        return player, llm_task, tts_task
