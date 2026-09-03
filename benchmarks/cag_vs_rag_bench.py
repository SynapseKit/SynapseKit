"""Benchmark comparing RAG (retrieval + generation) vs CAG (Cache-Augmented Generation).

Reports RAG vs CAG cold and CAG warm latencies separately (p50/p95).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from typing import Any
from unittest.mock import MagicMock

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.llm.llamacpp import LlamaCppLLM
from synapsekit.llm._llamacpp_cag_backend import LlamaCppCAGBackend
from synapsekit.rag.cag_router import CAGRouter
from synapsekit.rag.kv_cache_store import KVCacheStore
from synapsekit.retrieval.full_context import FullContextRetriever
from synapsekit.retrieval.retriever import Retriever


class MockRetriever(Retriever):
    def __init__(self) -> None:
        self.docs: list[str] = []

    async def add(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        self.docs.extend(texts)

    async def retrieve(
        self, query: str, top_k: int = 5, metadata_filter: dict | None = None
    ) -> list[str]:
        return self.docs[:top_k]


async def run_benchmark(
    llm: BaseLLM,
    corpus: list[str],
    queries: list[str],
    runs: int,
) -> dict[str, Any]:
    # Initialize cache store and backend
    cache_store = KVCacheStore(".synapsekit_bench_cag_cache")
    backend = LlamaCppCAGBackend()

    # Clean cache first
    if os.path.exists(".synapsekit_bench_cag_cache"):
        import shutil
        shutil.rmtree(".synapsekit_bench_cag_cache", ignore_errors=True)
    os.makedirs(".synapsekit_bench_cag_cache", exist_ok=True)

    # 1. Measure RAG
    # We build a retriever and measure ingestion and generation latencies
    rag_retriever = MockRetriever()
    await rag_retriever.add(corpus)

    rag_latencies: list[float] = []
    for _ in range(runs):
        for query in queries:
            t0 = time.perf_counter()
            # Simulate RAG retrieval + generation
            context = await rag_retriever.retrieve(query, top_k=5)
            prompt = "\n\n".join(context) + "\n\nQuestion: " + query
            _ = await llm.generate(prompt)
            rag_latencies.append((time.perf_counter() - t0) * 1000)

    # 2. Measure CAG Cold (includes building + saving + loading + generating)
    cag_cold_latencies: list[float] = []
    corpus_text = "\n\n".join(corpus)

    for _ in range(runs):
        for query in queries:
            # We clear the filesystem cache so we trigger a build
            if os.path.exists(".synapsekit_bench_cag_cache"):
                import shutil
                shutil.rmtree(".synapsekit_bench_cag_cache", ignore_errors=True)
            os.makedirs(".synapsekit_bench_cag_cache", exist_ok=True)

            t0 = time.perf_counter()
            # Build
            cache_handle = await backend.build_cache(llm, corpus_text)
            # Generate (uses load internally in generate_with_cache)
            tokens = []
            async for token in backend.generate_with_cache(llm, cache_handle, query):
                tokens.append(token)
            cag_cold_latencies.append((time.perf_counter() - t0) * 1000)

    # 3. Measure CAG Warm (uses pre-built and pre-saved cache)
    # First, build and save the cache once
    cache_handle = await backend.build_cache(llm, corpus_text)
    
    cag_warm_latencies: list[float] = []
    for _ in range(runs):
        for query in queries:
            t0 = time.perf_counter()
            # Generate (loads internally from cache_handle)
            tokens = []
            async for token in backend.generate_with_cache(llm, cache_handle, query):
                tokens.append(token)
            cag_warm_latencies.append((time.perf_counter() - t0) * 1000)

    # Clean up
    if os.path.exists(".synapsekit_bench_cag_cache"):
        import shutil
        shutil.rmtree(".synapsekit_bench_cag_cache", ignore_errors=True)

    return {
        "rag": rag_latencies,
        "cag_cold": cag_cold_latencies,
        "cag_warm": cag_warm_latencies,
    }


def report(results: dict[str, list[float]]) -> None:
    print(f"\n{'Approach':<12} | {'p50 (ms)':>10} | {'p95 (ms)':>10}")
    print("-" * 40)
    for name, latencies in results.items():
        if not latencies:
            continue
        p50 = statistics.median(latencies)
        latencies_sorted = sorted(latencies)
        p95 = latencies_sorted[min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)]
        print(f"{name:<12} | {p50:>10.2f} | {p95:>10.2f}")


class StubLlamaCppLLM(LlamaCppLLM):
    """Stub LLM for running benchmark when llama-cpp-python / GGUF model is not available."""

    def __init__(self) -> None:
        super().__init__(
            LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp"),
            model_path="/models/stub.gguf",
        )
        self._model = MagicMock()
        self._model.tokenize.return_value = [1, 2, 3]
        self._model.save_state.return_value = b"mock state"
        self._model.create_completion.return_value = [
            {"choices": [{"text": "Hello"}]},
            {"choices": [{"text": " world"}]},
        ]

    def _get_model(self) -> Any:
        return self._model


async def main_async(model_path: str | None, runs: int) -> None:
    if model_path:
        llm = LlamaCppLLM(
            LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp"),
            model_path=model_path,
        )
    else:
        print("No model path provided. Running with StubLlamaCppLLM.")
        llm = StubLlamaCppLLM()

    corpus = [
        "SynapseKit is an async-native framework for agent workflows.",
        "It supports building RAG pipelines, graph checkpointers, and advanced caching.",
    ]
    queries = [
        "What is SynapseKit?",
        "What does it support?",
    ]

    print(f"Running benchmarks ({runs} runs)...")
    results = await run_benchmark(llm, corpus, queries, runs)
    report(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Path to a GGUF model file.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per query.")
    args = parser.parse_args()

    asyncio.run(main_async(args.model_path, args.runs))


if __name__ == "__main__":
    main()
