"""
Multi-Provider LLM Comparison
==============================

This example demonstrates how to use the same prompt across multiple
LLM providers (OpenAI, Anthropic, Ollama) with SynapseKit.

Prerequisites:
    pip install synapsekit[openai,anthropic]
    # For Ollama: install ollama locally and pull models

Usage:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/multi_provider.py
"""

import asyncio
import os

from synapsekit import BaseLLM


# Mock provider configurations (replace with actual imports when available)
class OpenAIProvider:
    def __init__(self, api_key: str):
        from synapsekit.llm.openai import OpenAILLM
        self.llm = OpenAILLM(model="gpt-4o-mini", api_key=api_key)
    
    async def generate(self, prompt: str) -> str:
        response = await self.llm.agenerate(prompt)
        return response


class AnthropicProvider:
    def __init__(self, api_key: str):
        try:
            from synapsekit.llm.anthropic import AnthropicLLM
            self.llm = AnthropicLLM(model="claude-3-haiku-20240307", api_key=api_key)
        except ImportError:
            self.llm = None
    
    async def generate(self, prompt: str) -> str:
        if not self.llm:
            return "[Anthropic not configured]"
        response = await self.llm.agenerate(prompt)
        return response


class OllamaProvider:
    def __init__(self):
        try:
            from synapsekit.llm.ollama import OllamaLLM
            self.llm = OllamaLLM(model="llama2")
        except ImportError:
            self.llm = None
    
    async def generate(self, prompt: str) -> str:
        if not self.llm:
            return "[Ollama not configured]"
        response = await self.llm.agenerate(prompt)
        return response


async def compare_providers(prompt: str):
    """Compare responses from different providers"""
    
    print(f"Prompt: {prompt}\n")
    print("="*80)
    
    # OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print("\n🤖 OpenAI (GPT-4o-mini)")
        print("-"*80)
        openai = OpenAIProvider(openai_key)
        try:
            response = await openai.generate(prompt)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
    
    # Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("\n🧠 Anthropic (Claude 3 Haiku)")
        print("-"*80)
        anthropic = AnthropicProvider(anthropic_key)
        try:
            response = await anthropic.generate(prompt)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
    
    # Ollama (local)
    print("\n🦙 Ollama (Llama 2)")
    print("-"*80)
    ollama = OllamaProvider()
    try:
        response = await ollama.generate(prompt)
        print(response)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*80)


async def main():
    # Check environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Set it to enable OpenAI comparison.")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set. Set it to enable Anthropic comparison.")
    
    print()
    
    # Test prompts
    prompts = [
        "Explain quantum computing in one sentence.",
        "Write a haiku about artificial intelligence.",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'#'*80}")
        print(f"# Test {i}")
        print(f"{'#'*80}\n")
        await compare_providers(prompt)


if __name__ == "__main__":
    asyncio.run(main())
