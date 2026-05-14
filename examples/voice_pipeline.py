"""
Voice pipeline example — microphone → VAD → STT → LLM → TTS → speaker.

Usage
-----
    # With OpenAI providers (cloud STT + TTS):
    python examples/voice_pipeline.py

    # With local Whisper + OpenAI TTS:
    python examples/voice_pipeline.py --stt local

    # With local Whisper + Piper TTS (fully offline):
    python examples/voice_pipeline.py --stt local --tts piper --piper-model /path/to/model.onnx

Requirements
------------
    pip install 'synapsekit[voice,voice-stream]'

    # For local STT:
    pip install faster-whisper

    # For local TTS:
    pip install piper-tts
"""

from __future__ import annotations

import argparse
import asyncio
import os

from synapsekit.voice import (
    EnergyVAD,
    LocalWhisperSTT,
    OpenAITTS,
    OpenAIWhisperSTT,
    VoicePipeline,
)
from synapsekit.voice.types import PipelineEvent, PipelineState


# ── Event callback ─────────────────────────────────────────────────────────────

async def on_event(event: PipelineEvent) -> None:
    """Print pipeline events to the terminal in real time."""
    if event.kind == "state_change":
        state = event.data
        icons = {
            PipelineState.IDLE: "💤",
            PipelineState.LISTENING: "🎤",
            PipelineState.TRANSCRIBING: "📝",
            PipelineState.GENERATING: "🤔",
            PipelineState.SPEAKING: "🔊",
            PipelineState.INTERRUPTED: "✋",
        }
        icon = icons.get(state, "?")
        print(f"\r{icon} [{state.value.upper()}]          ", end="", flush=True)

    elif event.kind == "transcript":
        print(f"\nYou: {event.data}", flush=True)

    elif event.kind == "response_token":
        print(event.data, end="", flush=True)

    elif event.kind == "error":
        print(f"\n[ERROR] {event.data}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SynapseKit real-time voice pipeline demo")
    p.add_argument("--stt", choices=["openai", "local"], default="openai",
                   help="STT provider (default: openai)")
    p.add_argument("--tts", choices=["openai", "piper"], default="openai",
                   help="TTS provider (default: openai)")
    p.add_argument("--piper-model", default=None,
                   help="Path to Piper .onnx model (required when --tts piper)")
    p.add_argument("--whisper-model", default="base",
                   help="Whisper model size for local STT (default: base)")
    p.add_argument("--voice", default="alloy",
                   help="OpenAI TTS voice (default: alloy)")
    p.add_argument("--vad-threshold", type=float, default=0.01,
                   help="EnergyVAD threshold (default: 0.01)")
    p.add_argument("--interrupt-ms", type=int, default=300,
                   help="Interruption debounce in ms (default: 300)")
    p.add_argument("--system-prompt", default="You are a helpful voice assistant. Keep answers concise.",
                   help="System prompt for the assistant")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")

    # ── LLM ───────────────────────────────────────────────────────────────────
    try:
        from synapsekit.llm.base import LLMConfig
        from synapsekit.llm.openai import OpenAILLM

        llm = OpenAILLM(LLMConfig(model="gpt-4o-mini", api_key=api_key or "", provider="openai"))
    except ImportError:
        raise SystemExit(
            "openai is required. Install: pip install 'synapsekit[openai]'"
        )

    # ── STT ───────────────────────────────────────────────────────────────────
    if args.stt == "local":
        print(f"[STT] LocalWhisperSTT — model={args.whisper_model}")
        print("      Runs on-device (no network). First run downloads the model.")
        stt = LocalWhisperSTT(model=args.whisper_model)
    else:
        print("[STT] OpenAI Whisper API")
        stt = OpenAIWhisperSTT(api_key=api_key)

    # ── TTS ───────────────────────────────────────────────────────────────────
    if args.tts == "piper":
        if not args.piper_model:
            raise SystemExit("--piper-model is required when using --tts piper")
        from synapsekit.voice import PiperTTS

        print(f"[TTS] PiperTTS — model={args.piper_model} (fully offline)")
        tts = PiperTTS(model_path=args.piper_model)
        tts_sample_rate = 22050
    else:
        print(f"[TTS] OpenAI TTS — voice={args.voice}")
        tts = OpenAITTS(api_key=api_key, voice=args.voice)
        tts_sample_rate = 24000  # OpenAI PCM output is 24 kHz

    # ── VAD ───────────────────────────────────────────────────────────────────
    vad = EnergyVAD(threshold=args.vad_threshold)
    print(f"[VAD] EnergyVAD — threshold={args.vad_threshold}")
    print(
        "      Tip: use SileroVAD for noisy environments "
        "(pip install torch && from synapsekit.voice import SileroVAD)."
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = VoicePipeline(
        llm=llm,
        stt=stt,
        tts=tts,
        vad=vad,
        allow_interruption=True,
        interrupt_threshold_ms=args.interrupt_ms,
    )

    print("\n" + "=" * 60)
    print("  SynapseKit Real-Time Voice Pipeline")
    print("=" * 60)
    print(f"  System prompt : {args.system_prompt[:60]}...")
    print(f"  Interruption  : enabled — threshold {args.interrupt_ms} ms")
    print("  Press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    try:
        await pipeline.run_microphone(
            system_prompt=args.system_prompt,
            tts_sample_rate=tts_sample_rate,
            on_event=on_event,
        )
    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    asyncio.run(main())
