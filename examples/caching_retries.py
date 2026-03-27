"""
LLM with Caching and Retry Configuration
=========================================

This example demonstrates how to configure LLM calls with:
1. Response caching for repeated queries
2. Automatic retries with exponential backoff
3. Cost tracking and budget limits

Prerequisites:
    pip install synapsekit[openai]

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/caching_retries.py
"""

import asyncio
import os
from datetime import datetime

from synapsekit import BaseLLM, LLMConfig, CostTracker, BudgetGuard


# Simple in-memory cache implementation
class SimpleCache:
    """Simple in-memory cache for demonstration"""
    
    def __init__(self):
        self.cache = {}
    
    def get(self, key: str):
        """Get cached value"""
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        """Set cached value"""
        self.cache[key] = value
        print(f"💾 Cached response for: {key[:50]}...")
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        print("🗑️  Cache cleared")


async def demo_caching():
    """Demonstrate response caching"""
    print("\n" + "="*60)
    print("Demo 1: Response Caching")
    print("="*60 + "\n")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping demo.")
        return
    
    from synapsekit.llm.openai import OpenAILLM
    
    llm = OpenAILLM(model="gpt-4o-mini", api_key=api_key)
    cache = SimpleCache()
    
    prompt = "What is the capital of France?"
    
    # First call (no cache)
    print(f"Query: {prompt}")
    start = datetime.now()
    
    cached = cache.get(prompt)
    if cached:
        print("✅ Using cached response")
        response = cached
    else:
        print("🔄 Making API call...")
        response = await llm.agenerate(prompt)
        cache.set(prompt, response)
    
    elapsed = (datetime.now() - start).total_seconds()
    print(f"Response: {response}")
    print(f"⏱️  Time: {elapsed:.2f}s\n")
    
    # Second call (cached)
    print(f"Query (repeated): {prompt}")
    start = datetime.now()
    
    cached = cache.get(prompt)
    if cached:
        print("✅ Using cached response")
        response = cached
    else:
        print("🔄 Making API call...")
        response = await llm.agenerate(prompt)
        cache.set(prompt, response)
    
    elapsed = (datetime.now() - start).total_seconds()
    print(f"Response: {response}")
    print(f"⏱️  Time: {elapsed:.2f}s (much faster!)\n")


async def demo_retries():
    """Demonstrate retry configuration"""
    print("\n" + "="*60)
    print("Demo 2: Retry Configuration")
    print("="*60 + "\n")
    
    print("🔧 LLM Config with retries:")
    print("  - Max retries: 3")
    print("  - Timeout: 30s")
    print("  - Exponential backoff\n")
    
    # In a real scenario, configure via LLMConfig
    config = LLMConfig(
        max_retries=3,
        timeout=30,
        retry_on_errors=["RateLimitError", "TimeoutError"],
    )
    
    print(f"Config: {config}\n")
    print("💡 On API failures, requests will retry automatically with backoff.")


async def demo_cost_tracking():
    """Demonstrate cost tracking and budgets"""
    print("\n" + "="*60)
    print("Demo 3: Cost Tracking & Budget Limits")
    print("="*60 + "\n")
    
    # Initialize cost tracker
    tracker = CostTracker()
    
    # Simulate some API calls
    tracker.add_tokens(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
    tracker.add_tokens(model="gpt-4o-mini", prompt_tokens=200, completion_tokens=75)
    tracker.add_tokens(model="gpt-4o-mini", prompt_tokens=150, completion_tokens=60)
    
    print("📊 Cost Summary:")
    print(f"  Total tokens: {tracker.total_tokens}")
    print(f"  Total cost: ${tracker.total_cost:.4f}\n")
    
    # Set up budget guard
    print("🛡️  Setting budget limit: $5.00")
    budget = BudgetGuard(limit=5.0)
    
    try:
        # Check if under budget
        current_cost = tracker.total_cost
        if budget.check(current_cost):
            print(f"✅ Under budget (${current_cost:.4f} / $5.00)")
        else:
            print(f"❌ Budget exceeded!")
    except Exception as e:
        print(f"⚠️  {e}")


async def main():
    print("""
This example demonstrates three key LLM configuration patterns:
1. Response caching for repeated queries
2. Automatic retry with exponential backoff
3. Cost tracking and budget limits
    """)
    
    await demo_caching()
    await demo_retries()
    await demo_cost_tracking()
    
    print("\n✨ All demos complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
