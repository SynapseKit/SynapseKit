"""Real end-to-end RAG answered by Claude, streamed live to the dashboard.

    export ANTHROPIC_API_KEY=sk-ant-...        # your key (never hardcoded)
    python examples/live_rag_claude.py

Opens http://127.0.0.1:7900 and runs a real retrieval-augmented pipeline on a
loop: embed a small knowledge base → vector-search the question → build a
grounded prompt → **call Claude for real** → print the answer. Every step
(embeddings, retrieval, LLM) streams into SynapseKit Live automatically, and the
server stays up as long as this runs (Ctrl+C to stop).

Uses Haiku with small max_tokens, so each question costs a fraction of a cent.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys

import numpy as np

from synapsekit.embeddings.backend import SynapsekitEmbeddings
from synapsekit.live import enable
from synapsekit.live.instrument import _patch
from synapsekit.llm.anthropic import AnthropicLLM
from synapsekit.llm.base import LLMConfig
from synapsekit.retrieval.vectorstore import InMemoryVectorStore

KEY = os.environ.get("ANTHROPIC_API_KEY")
if not KEY:
    sys.exit("Set your key first:  export ANTHROPIC_API_KEY=sk-ant-...")

KNOWLEDGE_BASE = [
    "Refunds are processed within 5 business days to the original payment method.",
    "Orders over $50 ship free. Standard shipping takes 3 to 5 days.",
    "You can return any item within 30 days of delivery for a full refund.",
    "Gift cards are non-refundable and cannot be exchanged for cash.",
    "To track an order, use the tracking link in your confirmation email.",
]

QUESTIONS = [
    "How long do refunds take?",
    "Can I return something 3 weeks after it arrived?",
    "Are gift cards refundable?",
]

_DIM = 64


class BagOfWordsEmbeddings(SynapsekitEmbeddings):
    """Hashing bag-of-words — real lexical retrieval with no model download."""

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), _DIM), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vecs[row, hash(word) % _DIM] += 1.0
            norm = np.linalg.norm(vecs[row]) or 1.0
            vecs[row] /= norm
        return vecs


async def main() -> None:
    enable(open_browser=True)
    _patch(BagOfWordsEmbeddings, "embed", "embeddings.embed", lambda self, a, k: {"dim": _DIM})

    embeddings = BagOfWordsEmbeddings()
    store = InMemoryVectorStore(embedding_backend=embeddings)
    await store.add(KNOWLEDGE_BASE)

    llm = AnthropicLLM(
        LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            api_key=KEY,
            max_tokens=120,
            temperature=0,
        )
    )

    print("Real RAG + Claude, streaming to http://127.0.0.1:7900  (Ctrl+C to stop)\n")
    while True:
        for question in QUESTIONS:
            hits = await store.search(question, top_k=2)  # → retrieval event
            context = "\n".join(getattr(h, "text", str(h)) for h in hits)
            prompt = (
                "Answer the customer's question using ONLY this policy context.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            answer = await llm.generate(prompt)  # → real Claude call (llm event)
            print(f"Q: {question}\nA: {answer.strip()}\n")
            await asyncio.sleep(1.5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
