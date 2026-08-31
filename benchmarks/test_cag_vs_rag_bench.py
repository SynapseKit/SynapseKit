from __future__ import annotations

import sys
from pathlib import Path

# Add benchmarks directory and repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cag_vs_rag_bench import run_benchmark, StubLlamaCppLLM


@pytest.mark.asyncio
async def test_cag_vs_rag_bench_harness() -> None:
    llm = StubLlamaCppLLM()
    corpus = ["This is the first sentence.", "This is the second sentence."]
    queries = ["What is first?", "What is second?"]
    
    results = await run_benchmark(llm, corpus, queries, runs=1)
    
    assert "rag" in results
    assert "cag_cold" in results
    assert "cag_warm" in results
    
    assert len(results["rag"]) == 2
    assert len(results["cag_cold"]) == 2
    assert len(results["cag_warm"]) == 2
    
    for val in results["rag"]:
        assert val >= 0
