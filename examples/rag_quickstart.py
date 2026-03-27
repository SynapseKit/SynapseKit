"""
RAG Quickstart Example
======================

This example demonstrates the simplest way to get started with SynapseKit's RAG system:
1. Initialize RAG with a model and API key
2. Add text documents
3. Query and stream responses

Prerequisites:
    pip install synapsekit[openai]

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/rag_quickstart.py
"""

import asyncio
import os

from synapsekit import RAG


async def main():
    # Initialize RAG with your preferred model
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    rag = RAG(model="gpt-4o-mini", api_key=api_key)

    # Add some documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "SynapseKit is a lightweight, async-first RAG framework for building AI applications.",
    ]

    for doc in documents:
        rag.add(doc)

    print("Documents added to RAG system.\n")

    # Query the RAG system with streaming
    query = "What is SynapseKit?"
    print(f"Query: {query}\n")
    print("Response: ", end="", flush=True)

    async for token in rag.stream(query):
        print(token, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
