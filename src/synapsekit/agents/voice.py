"""VoiceAgent — orchestrates STT, inner agent, and TTS pipelines."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm.base import BaseLLM
from .base import BaseTool
from .function_calling import FunctionCallingAgent
from .react import ReActAgent
from .voice_backends import (
    EnergyVAD,
    OpenAITTSBackend,
    Pyttsx3TTSBackend,
    STTBackend,
    TTSBackend,
    WhisperAPIBackend,
    WhisperLocalBackend,
)


@dataclass
class VoiceResult:
    """Result of a VoiceAgent execution."""

    transcript: str
    response: str
    audio: bytes | None
    output_path: str | None


class VoiceAgent:
    """Real-time voice agent combining STT -> Agent -> TTS."""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        tools: list[BaseTool] | None = None,
        *,
        stt_model: str = "whisper-1",
        tts_model: str = "tts-1",
        tts_voice: str = "alloy",
        stt_backend: STTBackend | None = None,
        tts_backend: TTSBackend | None = None,
        vad: bool = False,
        vad_threshold: float = 0.01,
        agent: ReActAgent | FunctionCallingAgent | None = None,
        system_prompt: str = "You are a helpful voice assistant. Keep answers concise.",
        api_key: str | None = None,
        max_iterations: int = 10,
    ) -> None:
        if agent is None:
            if llm is None:
                raise ValueError("Either 'agent' or 'llm' must be provided.")
            self._agent: ReActAgent | FunctionCallingAgent = FunctionCallingAgent(
                llm=llm,
                tools=tools or [],
                max_iterations=max_iterations,
                system_prompt=system_prompt,
            )
        else:
            self._agent = agent

        # STT Backend logic
        if stt_backend is not None:
            self._stt = stt_backend
        elif stt_model.startswith("whisper-1"):
            self._stt = WhisperAPIBackend(model=stt_model, api_key=api_key)
        else:
            self._stt = WhisperLocalBackend(model=stt_model)

        # TTS Backend logic
        if tts_backend is not None:
            self._tts = tts_backend
        elif tts_model == "pyttsx3":
            self._tts = Pyttsx3TTSBackend(voice=tts_voice)
        else:
            self._tts = OpenAITTSBackend(model=tts_model, voice=tts_voice, api_key=api_key)

        self._vad = EnergyVAD(threshold=vad_threshold) if vad else None

    async def run_file(
        self,
        input_path: str | Path,
        *,
        output: str | Path | None = None,
        output_format: str = "mp3",
    ) -> VoiceResult:
        """Process an audio file, get an agent response, and optionally save audio."""
        if isinstance(input_path, str):
            input_path = Path(input_path)
        transcript = await self._stt.transcribe(input_path)
        if not transcript.strip():
            return VoiceResult(
                transcript="",
                response="[No speech detected]",
                audio=None,
                output_path=None,
            )

        response_text = await self._agent.run(transcript)

        audio_bytes = await self._tts.synthesize(response_text)

        out_path_str = None
        if output and audio_bytes:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(audio_bytes)
            out_path_str = str(out_path)

        return VoiceResult(
            transcript=transcript,
            response=response_text,
            audio=audio_bytes,
            output_path=out_path_str,
        )

    async def run_stream(
        self,
        *,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 30,
    ) -> None:
        """Stream mode: mic -> agent -> speaker."""
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "sounddevice and soundfile are required for run_stream. "
                "Install with: pip install 'synapsekit[voice-stream]'"
            ) from None

        import io
        import queue
        import sys

        q: queue.Queue = queue.Queue()

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))

        print("Listening... (Press Ctrl+C to stop)")

        audio_buffer = bytearray()
        silence_chunks = 0
        max_silence_chunks = int(1000 / chunk_duration_ms) * 2  # 2 seconds of silence
        is_speaking = False

        chunk_size = int(sample_rate * chunk_duration_ms / 1000)

        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=chunk_size,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                try:
                    if q.empty():
                        await asyncio.sleep(0.01)
                        continue

                    data = q.get_nowait()

                    if self._vad:
                        has_speech = self._vad.is_speech(data)
                    else:
                        has_speech = True

                    if has_speech:
                        if not is_speaking:
                            print("Speech detected...")
                        is_speaking = True
                        silence_chunks = 0
                        audio_buffer.extend(data)
                    elif is_speaking:
                        silence_chunks += 1
                        audio_buffer.extend(data)

                        if silence_chunks > max_silence_chunks:
                            print("Processing speech...")
                            wav_io = io.BytesIO()
                            with sf.SoundFile(
                                wav_io, mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16"
                            ) as file:
                                import numpy as np

                                float_data = (
                                    np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
                                    / 32768.0
                                )
                                file.write(float_data)

                            wav_bytes = wav_io.getvalue()

                            transcript = await self._stt.transcribe(wav_bytes)
                            print(f"User: {transcript}")

                            if transcript.strip():
                                response_text = await self._agent.run(transcript)
                                print(f"Agent: {response_text}")

                                audio_bytes = await self._tts.synthesize(response_text)
                                if audio_bytes:
                                    out_io = io.BytesIO(audio_bytes)
                                    try:
                                        playback_data, fs = sf.read(out_io)
                                        sd.play(playback_data, fs)
                                        sd.wait()
                                    except Exception as e:
                                        print(f"Error playing audio: {e}", file=sys.stderr)

                            audio_buffer.clear()
                            is_speaking = False
                            print("Listening...")

                except queue.Empty:
                    # Ignore queue.Empty; wait for next loop tick.
                    continue
                except KeyboardInterrupt:
                    print("Stopping stream.")
                    break

    async def ws_handler(self, websocket: Any) -> None:
        """
        WebSocket mode for integrating with web apps.
        Expects binary frames containing audio from client, sends binary audio responses.
        """
        audio_buffer = bytearray()

        async def _process_buffer() -> None:
            if not audio_buffer:
                return
            wav_bytes = bytes(audio_buffer)
            audio_buffer.clear()

            transcript = await self._stt.transcribe(wav_bytes)
            if transcript.strip():
                response_text = await self._agent.run(transcript)
                audio_bytes = await self._tts.synthesize(response_text)

                if hasattr(websocket, "send_json"):
                    await websocket.send_json(
                        {"transcript": transcript, "response": response_text}
                    )
                elif hasattr(websocket, "send_text"):
                    import json

                    await websocket.send_text(
                        json.dumps(
                            {"transcript": transcript, "response": response_text}
                        )
                    )
                elif hasattr(websocket, "send"):
                    import json

                    await websocket.send(
                        json.dumps(
                            {"transcript": transcript, "response": response_text}
                        )
                    )

                if audio_bytes:
                    if hasattr(websocket, "send_bytes"):
                        await websocket.send_bytes(audio_bytes)
                    elif hasattr(websocket, "send"):
                        await websocket.send(audio_bytes)

        try:
            if hasattr(websocket, "iter_bytes"):
                # e.g., FastAPI/Starlette
                iterator = websocket.iter_bytes()
            else:
                # e.g., websockets
                iterator = websocket

            async for message in iterator:
                if isinstance(message, bytes):
                    audio_buffer.extend(message)

                    if self._vad and len(audio_buffer) > 16000:
                        chunk = audio_buffer[-16000:]
                        if not self._vad.is_speech(chunk):
                            await _process_buffer()

                elif isinstance(message, str):
                    if message == "stop":
                        await _process_buffer()
                        break
        except Exception:
            # Handle disconnects
            pass
        finally:
            await _process_buffer()
