"""Reasoning models example with openai, anthropic, google, deepseek and qwen.

This example demonstrates how to use ReasoningLLM across different providers.

Usage:
    python examples/reasoning_models.py

Requirements:
    Set API keys via environment variables or pass them explicitly.
"""

from __future__ import annotations

import asyncio
import os


async def main():
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""

    print("=" * 60)
    print("ReasoningLLM - Unified API for Reasoning Models")
    print("=" * 60)

    # -------------------------
    # OpenAI (o3 / o1)
    # -------------------------
    print("\n--- OpenAI o3 ---")
    try:
        from synapsekit import ReasoningLLM

        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            llm = ReasoningLLM(model="o3", api_key=openai_key)
            result = await llm.agenerate(
                "Explain what happens in a transformer when computing attention"
            )

            print(f"Model: {result.model}")
            print(f"Provider: {result.provider}")
            print(f"Thinking tokens: {result.thinking_tokens}")
            print(f"Answer tokens: {result.answer_tokens}")
            print(f"Total tokens: {result.total_tokens}")

            if result.thinking:
                print(f"\nThinking (truncated): {result.thinking[:200]}...")
            print(f"\nAnswer: {result.answer}")
        else:
            print("Skipped: OPENAI_API_KEY not set")

    except Exception as e:
        print(f"Error: {e}")

    # -------------------------
    # Anthropic (Claude with thinking)
    # -------------------------
    print("\n--- Anthropic Claude ---")
    try:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            llm = ReasoningLLM(
                model="claude-sonnet-4-6-20251001",
                api_key=anthropic_key,
                budget_tokens=1024,
            )
            result = await llm.agenerate("Explain the chain of thought reasoning in transformers")

            print(f"Model: {result.model}")
            print(f"Provider: {result.provider}")
            print(f"Thinking tokens: {result.thinking_tokens}")
            print(f"Answer tokens: {result.answer_tokens}")

            if result.thinking:
                print(f"\nThinking (truncated): {result.thinking[:200]}...")
            print(f"\nAnswer: {result.answer}")
        else:
            print("Skipped: ANTHROPIC_API_KEY not set")

    except Exception as e:
        print(f"Error: {e}")

    # -------------------------
    # Google Gemini (with thinking)
    # -------------------------
    print("\n--- Google Gemini ---")
    try:
        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key:
            llm = ReasoningLLM(
                model="gemini-2.5-pro-preview-0506",
                api_key=google_key,
                budget_tokens=1024,
            )
            result = await llm.agenerate("Explain self-attention mechanism in transformers")

            print(f"Model: {result.model}")
            print(f"Provider: {result.provider}")
            print(f"Thinking tokens: {result.thinking_tokens}")
            print(f"Answer tokens: {result.answer_tokens}")

            if result.thinking:
                print(f"\nThinking (truncated): {result.thinking[:200]}...")
            print(f"\nAnswer: {result.answer}")
        else:
            print("Skipped: GOOGLE_API_KEY not set")

    except Exception as e:
        print(f"Error: {e}")

    # -------------------------
    # DeepSeek R1
    # -------------------------
    print("\n--- DeepSeek R1 ---")
    try:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key:
            llm = ReasoningLLM(model="deepseek-reasoner", api_key=deepseek_key)
            result = await llm.agenerate("Explain the mathematical formulation of attention")

            print(f"Model: {result.model}")
            print(f"Provider: {result.provider}")
            print(f"Thinking tokens: {result.thinking_tokens}")
            print(f"Answer tokens: {result.answer_tokens}")

            if result.thinking:
                print(f"\nThinking (truncated): {result.thinking[:200]}...")
            print(f"\nAnswer: {result.answer}")
        else:
            print("Skipped: DEEPSEEK_API_KEY not set")

    except Exception as e:
        print(f"Error: {e}")

    # -------------------------
    # Qwen QwQ
    # -------------------------
    print("\n--- Qwen QwQ ---")
    try:
        qwen_key = os.environ.get("DASHSCOPE_API_KEY")
        if qwen_key:
            llm = ReasoningLLM(model="qwq-32b", api_key=qwen_key)
            result = await llm.agenerate("Explain how transformers process sequential data")

            print(f"Model: {result.model}")
            print(f"Provider: {result.provider}")
            print(f"Thinking tokens: {result.thinking_tokens}")
            print(f"Answer tokens: {result.answer_tokens}")

            if result.thinking:
                print(f"\nThinking (truncated): {result.thinking[:200]}...")
            print(f"\nAnswer: {result.answer}")
        else:
            print("Skipped: DASHSCOPE_API_KEY not set")

    except Exception as e:
        print(f"Error: {e}")

    # -------------------------
    # Streaming Example
    # -------------------------
    print("\n--- Streaming Example ---")
    try:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            llm = ReasoningLLM(model="o3", api_key=openai_key)

            print("Streaming thinking + answer:")
            async for chunk in llm.astream("Explain addition"):
                marker = "[THINK] " if chunk.is_thinking else "[ANSWER] "
                print(f"{marker}{chunk.text}", end="", flush=True)

            print("\n\nDone")
        else:
            print("Skipped: OPENAI_API_KEY not set")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("ReasoningLLM example complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
