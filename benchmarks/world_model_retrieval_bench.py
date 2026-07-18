"""Hybrid vs. vector-only retrieval accuracy comparison for WorldModelRAG.

Measures hit@k and MRR on a synthetic multi-hop question set (see
``examples/world_model_corpus.py``) for two retrieval strategies:

* **hybrid** -- ``WorldModelRAG``'s graph+vector fusion
  (``HybridWorldModelRetriever.retrieve_with_scores(strategy="hybrid")``).
* **vector-only** -- the underlying vector retriever directly
  (``wm.vector_retriever.retrieve_with_scores``), bypassing the graph
  entirely. This is the true baseline: none of the retriever's own
  strategies ("graph_first", "vector_first", "hybrid") is graph-free.

Each question names only the head entity (person) of a causal chain and
asks what resulted from their work; the ground-truth relevant document is
the *tail* of the chain (the incident), which shares no vocabulary with the
question and is only reachable by following the causal edge(s) -- so this
is a structural test of graph-assisted retrieval, not an incidental one.

This measures retrieval *accuracy*, not speed, so it's a standalone script
rather than part of the ``benchmarks/pytest.ini`` wall-clock-timing suite.
See ``test_world_model_retrieval_bench.py`` for the CI-enforced regression
check.

Usage:
    python benchmarks/world_model_retrieval_bench.py --n-docs 10000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from world_model_corpus import Chain, generate_corpus, question_for_chain

from synapsekit import WorldModelRAG
from synapsekit.mesh.embeddings import HashingEmbeddings
from synapsekit.retrieval.vectorstore import InMemoryVectorStore
from synapsekit.retrieval.world_model import HeuristicWorldModelExtractor


def hit_at_k(doc_ids: list[str], relevant: set[str], k: int) -> bool:
    return any(doc_id in relevant for doc_id in doc_ids[:k])


def reciprocal_rank(doc_ids: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


async def build_world_model(docs: list[dict]) -> WorldModelRAG:
    wm = WorldModelRAG(
        graph_backend="in_memory",
        extractor=HeuristicWorldModelExtractor(),
        vector_store=InMemoryVectorStore(HashingEmbeddings()),
        llm=None,
    )
    await wm.ingest(docs)
    return wm


async def compare(
    wm: WorldModelRAG, chains: list[Chain], top_k: int
) -> dict[str, dict[str, float]]:
    """Return per-strategy mean hit@k and MRR over the chain question set."""
    hits: dict[str, list[bool]] = {"hybrid": [], "vector_only": []}
    rr: dict[str, list[float]] = {"hybrid": [], "vector_only": []}

    for chain in chains:
        question = question_for_chain(chain)
        relevant = {chain.tail_doc_id}

        hybrid = await wm.retriever.retrieve_with_scores(question, top_k=top_k, strategy="hybrid")
        vector_only = await wm.vector_retriever.retrieve_with_scores(question, top_k=top_k)

        hybrid_ids = [str(r["metadata"]["source"]) for r in hybrid]
        vector_ids = [str(r["metadata"]["source"]) for r in vector_only]

        hits["hybrid"].append(hit_at_k(hybrid_ids, relevant, top_k))
        hits["vector_only"].append(hit_at_k(vector_ids, relevant, top_k))
        rr["hybrid"].append(reciprocal_rank(hybrid_ids, relevant))
        rr["vector_only"].append(reciprocal_rank(vector_ids, relevant))

    n = len(chains)
    return {
        strategy: {
            "hit_rate": sum(hits[strategy]) / n,
            "mrr": sum(rr[strategy]) / n,
        }
        for strategy in ("hybrid", "vector_only")
    }


def relative_improvement(hybrid: float, baseline: float) -> float:
    """Relative improvement of hybrid over baseline; treats a 0 baseline as +inf."""
    if baseline == 0:
        return float("inf") if hybrid > 0 else 0.0
    return (hybrid - baseline) / baseline


async def run(n_docs: int, n_questions: int | None, top_k: int, seed: int) -> None:
    docs, chains = generate_corpus(n_docs=n_docs, seed=seed)
    if n_questions is not None:
        chains = chains[:n_questions]

    t0 = time.time()
    wm = await build_world_model(docs)
    ingest_s = time.time() - t0

    t0 = time.time()
    results = await compare(wm, chains, top_k)
    query_s = time.time() - t0

    print(f"corpus: {n_docs} docs, {len(chains)} questions, top_k={top_k}, seed={seed}")
    print(f"ingest: {ingest_s:.1f}s, retrieval: {query_s:.1f}s")
    print()
    print(f"{'strategy':<14}{'hit@k':>10}{'mrr':>10}")
    for strategy in ("hybrid", "vector_only"):
        r = results[strategy]
        print(f"{strategy:<14}{r['hit_rate']:>10.3f}{r['mrr']:>10.3f}")
    print()
    hit_improvement = relative_improvement(
        results["hybrid"]["hit_rate"], results["vector_only"]["hit_rate"]
    )
    mrr_improvement = relative_improvement(results["hybrid"]["mrr"], results["vector_only"]["mrr"])
    print(f"hybrid hit@k improvement over vector-only: {hit_improvement:+.1%}")
    print(f"hybrid mrr improvement over vector-only:   {mrr_improvement:+.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-docs", type=int, default=10_000)
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(run(args.n_docs, args.n_questions, args.top_k, args.seed))


if __name__ == "__main__":
    main()
