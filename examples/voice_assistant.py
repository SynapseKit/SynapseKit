"""
Voice assistant with persistent memory — complete example.

Demonstrates a production-grade voice loop with:
  * EnergyVAD or SileroVAD (--silero flag)
  * LocalWhisperSTT (default) or OpenAI Whisper API (--cloud-stt)
  * OpenAI TTS, ElevenLabs (--elevenlabs), or Cartesia (--cartesia)
  * AgentMemory for cross-session persistence (SQLite by default)
  * Interruption handling (300 ms debounce)
  * Sentence-level TTS streaming

Platform notes
--------------
  macOS:  Works out of the box. Microphone permission required on first run.
  Linux:  Requires PortAudio: ``sudo apt-get install libportaudio2``
  Windows: Works best-effort; PortAudio ships with the sounddevice wheel.

Quick start
-----------
    pip install 'synapsekit[voice]' faster-whisper
    export OPENAI_API_KEY=sk-...
    python examples/voice_assistant.py

With persistent memory across sessions::

    python examples/voice_assistant.py --memory sqlite --agent-id alice

With Cartesia TTS::

    pip install 'synapsekit[voice-cartesia]'
    python examples/voice_assistant.py --tts cartesia --cartesia-key <key>

With ElevenLabs TTS::

    pip install 'synapsekit[voice-elevenlabs]'
    python examples/voice_assistant.py --tts elevenlabs --elevenlabs-key <key>
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from synapsekit.voice import (
    EnergyVAD,
    LocalWhisperSTT,
    OpenAITTS,
    OpenAIWhisperSTT,
    VoicePipeline,
)
from synapsekit.voice.types import PipelineEvent, PipelineState

_STATE_LABEL = {
    PipelineState.IDLE: "idle",
    PipelineState.LISTENING: "listening ...",
    PipelineState.TRANSCRIBING: "transcribing ...",
    PipelineState.GENERATING: "thinking ...",
    PipelineState.SPEAKING: "speaking",
    PipelineState.INTERRUPTED: "interrupted",
}


async def on_event(event: PipelineEvent) -> None:
    if event.kind == "state_change" and isinstance(event.data, PipelineState):
        label = _STATE_LABEL.get(event.data, event.data.value)
        print(f"\r[{label}]          ", end="", flush=True)

    elif event.kind == "transcript":
        print(f"\n\nYou:       {event.data}", flush=True)
        print("Assistant: ", end="", flush=True)

    elif event.kind == "response_token":
        print(event.data, end="", flush=True)

    elif event.kind == "error":
        print(f"\n[error] {event.data}", file=sys.stderr)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SynapseKit voice assistant with persistent memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # STT
    p.add_argument("--stt", choices=["local", "openai"], default="local",
                   help="STT provider (default: local — runs on-device via faster-whisper)")
    p.add_argument("--whisper-model", default="base",
                   help="Whisper model size: tiny/base/small/medium/large-v3 (default: base)")

    # TTS
    p.add_argument("--tts", choices=["openai", "elevenlabs", "cartesia"], default="openai",
                   help="TTS provider (default: openai)")
    p.add_argument("--tts-voice", default="nova",
                   help="OpenAI TTS voice (default: nova)")
    p.add_argument("--elevenlabs-key", default=None,
                   help="ElevenLabs API key (required when --tts elevenlabs)")
    p.add_argument("--elevenlabs-voice", default="21m00Tcm4TlvDq8ikWAM",
                   help="ElevenLabs voice ID")
    p.add_argument("--cartesia-key", default=None,
                   help="Cartesia API key (required when --tts cartesia)")
    p.add_argument("--cartesia-voice", default="a0e99841-438c-4a64-b679-ae501e7d6091",
                   help="Cartesia voice ID")

    # VAD
    p.add_argument("--silero", action="store_true",
                   help="Use SileroVAD instead of EnergyVAD (requires torch)")
    p.add_argument("--vad-threshold", type=float, default=0.01,
                   help="EnergyVAD threshold 0-1 (default 0.01)")

    # Memory
    p.add_argument("--memory", choices=["none", "memory", "sqlite"], default="sqlite",
                   help="Memory backend (default: sqlite — persists across sessions)")
    p.add_argument("--memory-db", default="voice_assistant_memory.db",
                   help="SQLite file for persistent memory (default: voice_assistant_memory.db)")
    p.add_argument("--agent-id", default="default",
                   help="Agent identity for memory namespacing (default: default)")
    p.add_argument("--memory-top-k", type=int, default=3,
                   help="Memories recalled per turn (default: 3)")

    # Pipeline
    p.add_argument("--interrupt-ms", type=int, default=300,
                   help="Interruption debounce in ms (default: 300)")
    p.add_argument("--silence-ms", type=int, default=1500,
                   help="Silence required to finalize utterance in ms (default: 1500)")
    p.add_argument("--system-prompt", default=(
        "You are a helpful voice assistant. "
        "Keep answers concise and conversational — aim for 1-3 sentences."
    ))

    # LLM
    p.add_argument("--llm-model", default="gpt-4o-mini",
                   help="LLM model (default: gpt-4o-mini)")

    return p


async def main() -> None:
    args = _build_arg_parser().parse_args()
    openai_key = os.environ.get("OPENAI_API_KEY")

    # ── LLM ───────────────────────────────────────────────────────────────────
    try:
        from synapsekit.llm.openai import OpenAILLM
        from synapsekit.llm.base import LLMConfig
    except ImportError:
        raise SystemExit("Install openai: pip install 'synapsekit[openai]'")

    llm = OpenAILLM(LLMConfig(
        model=args.llm_model,
        api_key=openai_key or "",
        provider="openai",
    ))

    # ── STT ───────────────────────────────────────────────────────────────────
    if args.stt == "local":
        print(f"[stt] LocalWhisperSTT  model={args.whisper_model}  (on-device, no network)")
        stt = LocalWhisperSTT(model=args.whisper_model)
    else:
        print("[stt] OpenAI Whisper API  (network round-trip ~500 ms)")
        stt = OpenAIWhisperSTT(api_key=openai_key)

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_sample_rate = 24000  # default for OpenAI PCM

    if args.tts == "elevenlabs":
        from synapsekit.voice import ElevenLabsTTS

        key = args.elevenlabs_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not key:
            raise SystemExit("Set --elevenlabs-key or ELEVENLABS_API_KEY")
        print(f"[tts] ElevenLabs  voice={args.elevenlabs_voice}")
        tts = ElevenLabsTTS(api_key=key, voice_id=args.elevenlabs_voice)

    elif args.tts == "cartesia":
        from synapsekit.voice import CartesiaTTS

        key = args.cartesia_key or os.environ.get("CARTESIA_API_KEY", "")
        if not key:
            raise SystemExit("Set --cartesia-key or CARTESIA_API_KEY")
        print(f"[tts] Cartesia  voice={args.cartesia_voice}")
        tts = CartesiaTTS(api_key=key, voice_id=args.cartesia_voice, sample_rate=24000)

    else:
        print(f"[tts] OpenAI TTS  voice={args.tts_voice}")
        tts = OpenAITTS(api_key=openai_key, voice=args.tts_voice)

    # ── VAD ───────────────────────────────────────────────────────────────────
    if args.silero:
        from synapsekit.voice import SileroVAD

        print("[vad] SileroVAD  (neural, ~2 ms/frame)")
        vad = SileroVAD()
    else:
        print(f"[vad] EnergyVAD  threshold={args.vad_threshold}")
        vad = EnergyVAD(threshold=args.vad_threshold)

    # ── Memory ────────────────────────────────────────────────────────────────
    memory = None
    if args.memory != "none":
        from synapsekit.memory.agent_memory import AgentMemory

        if args.memory == "sqlite":
            print(f"[mem] SQLite  db={args.memory_db}  agent={args.agent_id}")
            memory = AgentMemory(backend="sqlite", path=args.memory_db, llm=llm)
        else:
            print(f"[mem] in-process  agent={args.agent_id}")
            memory = AgentMemory(backend="memory", llm=llm)
    else:
        print("[mem] disabled")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = VoicePipeline(
        llm=llm,
        stt=stt,
        tts=tts,
        vad=vad,
        allow_interruption=True,
        interrupt_threshold_ms=args.interrupt_ms,
        memory=memory,
        agent_id=args.agent_id,
        memory_top_k=args.memory_top_k,
    )

    print()
    print("=" * 60)
    print("  SynapseKit Voice Assistant")
    print("=" * 60)
    print(f"  LLM model  : {args.llm_model}")
    print(f"  Interrupts : enabled ({args.interrupt_ms} ms debounce)")
    if memory:
        print(f"  Memory     : {args.memory} / agent={args.agent_id}")
    print("  Speak naturally. Press Ctrl-C to exit.")
    print("=" * 60)
    print()

    try:
        await pipeline.run_microphone(
            system_prompt=args.system_prompt,
            silence_duration_ms=args.silence_ms,
            tts_sample_rate=tts_sample_rate,
            on_event=on_event,
        )
    except KeyboardInterrupt:
        print("\n\nGoodbye.")


if __name__ == "__main__":
    asyncio.run(main())
