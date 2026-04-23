"""Multimodal RAG example: add image/audio/video files to one pipeline.

Prerequisites:
    pip install synapsekit[openai]
    ffmpeg (required for video)

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/multimodal_rag.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from synapsekit import RAG


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    rag = RAG(model="gpt-4o-mini", api_key=api_key)

    image_path = Path("./data/architecture_diagram.png")
    audio_path = Path("./data/meeting.mp3")
    video_path = Path("./data/product_demo.mp4")

    if image_path.exists():
        await rag.add_async(str(image_path), caption="Authentication architecture diagram")

    if audio_path.exists():
        await rag.add_async(str(audio_path))

    if video_path.exists():
        await rag.add_async(str(video_path), frame_interval=30)

    question = "What does the authentication architecture look like?"
    answer = await rag.ask(question)
    print(f"Q: {question}\nA: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
