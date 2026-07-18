"""End-to-end WorldModelRAG demo at 10k-document scale.

Ingests a deterministic synthetic corpus (see ``world_model_corpus.py``) of
causal two-hop chains -- "Person worked on Product" / "Product caused
Incident" -- spread across separate documents, then asks a multi-hop
question that names only the person and requires graph traversal (not
vector similarity alone) to reach the resulting incident.

Runs fully offline by default: ``HeuristicWorldModelExtractor`` (regex-based,
zero-cost) and ``HashingEmbeddings`` (deterministic, numpy-only) mean no
network calls or API key are required. Ingesting and querying 10k documents
is pure CPU regex/hash work -- expect low tens of seconds. Set
``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY`` to have the final answer
synthesized by a real LLM instead of the canned offline stub.
"""

import asyncio
import os
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from world_model_corpus import generate_corpus, question_for_chain

from synapsekit import ExtractionPolicy, WorldModelRAG
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.mesh.embeddings import HashingEmbeddings
from synapsekit.retrieval.vectorstore import InMemoryVectorStore
from synapsekit.retrieval.world_model import HeuristicWorldModelExtractor


class DemoLLM(BaseLLM):
    """Offline stand-in used when no LLM API key is configured."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(model="demo", api_key="", provider="demo"))

    async def stream(self, prompt: str, **kw) -> AsyncGenerator[str]:
        yield (
            "Based on the retrieved graph and context above, the causal chain "
            "connects the named person's work to the incident that followed."
        )


def _llm_kwargs() -> dict:
    """Use a real provider if an API key is set in the environment, else offline."""
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        return {"model": "claude-3-5-haiku-20241022", "api_key": api_key}
    if api_key := os.environ.get("OPENAI_API_KEY"):
        return {"model": "gpt-4o-mini", "api_key": api_key}
    return {"llm": DemoLLM()}


async def main() -> None:
    docs, chains = generate_corpus(n_docs=10_000, seed=42)

    wm = WorldModelRAG(
        extraction=ExtractionPolicy(temporal=True, causal=True),
        graph_backend="in_memory",
        extractor=HeuristicWorldModelExtractor(),
        vector_store=InMemoryVectorStore(HashingEmbeddings()),
        retrieval_top_k=5,
        **_llm_kwargs(),
    )

    t0 = time.time()
    await wm.ingest(docs)
    print(f"Ingested {len(docs)} documents in {time.time() - t0:.1f}s")

    for chain in chains[:2]:
        question = question_for_chain(chain)
        result = await wm.query(question, strategy="hybrid")
        print()
        print(f"Q: {question}")
        print(f"A: {result.answer}")
        print(f"Subgraph documents: {result.subgraph.documents}")

    print()
    print(wm.subgraph_to_mermaid(question_for_chain(chains[0])))


if __name__ == "__main__":
    asyncio.run(main())
