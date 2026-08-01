"""Watch EVERY SynapseKit subsystem — plus a real Claude RAG answer — stream live.

    export ANTHROPIC_API_KEY=sk-ant-...     # optional: real Claude; else a local FakeLLM
    python examples/live_all_features.py

Opens http://127.0.0.1:7900 and loops through the whole framework with real
objects, auto-instrumented by ``synapsekit.live``:

    loader → embeddings → vector search → tool → MCP tool → memory (DB)
           → knowledge graph → RAG answer (real Claude if a key is set)

The server stays up as long as this runs (Ctrl+C to stop), so the browser tab
stays connected. With a key it uses Haiku + small max_tokens (a fraction of a
cent per loop).
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.embeddings.backend import SynapsekitEmbeddings
from synapsekit.live import enable
from synapsekit.live.instrument import _patch
from synapsekit.llm.base import BaseLLM, LLMConfig

_DIM = 64
KNOWLEDGE_BASE = [
    "Refunds are processed within 5 business days to the original payment method.",
    "Orders over $50 ship free. Standard shipping takes 3 to 5 days.",
    "You can return any item within 30 days of delivery for a full refund.",
    "Gift cards are non-refundable and cannot be exchanged for cash.",
]
QUESTION = "How long do refunds take?"


class BagOfWordsEmbeddings(SynapsekitEmbeddings):
    """Hashing bag-of-words — real lexical retrieval with no model download."""

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), _DIM), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vecs[row, hash(word) % _DIM] += 1.0
            vecs[row] /= np.linalg.norm(vecs[row]) or 1.0
        return vecs


class FakeLLM(BaseLLM):
    async def stream(self, prompt: str, **kw: object) -> AsyncGenerator[str, None]:
        for word in ("Refunds", "take", "5", "business", "days."):
            yield word + " "


class SearchDocsTool(BaseTool):
    name = "search_docs"
    description = "Search the docs."
    parameters = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        return ToolResult(output="found 3 docs")


def _hit_text(hit: object) -> str:
    if isinstance(hit, dict):
        return str(hit.get("text", hit))
    return str(getattr(hit, "text", hit))


def _make_llm() -> BaseLLM:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        from synapsekit.llm.anthropic import AnthropicLLM

        return AnthropicLLM(
            LLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                api_key=key,
                max_tokens=120,
                temperature=0,
            )
        )
    return FakeLLM(LLMConfig(provider="fake", model="fake-1", api_key=""))


async def _guard(label: str, fn) -> None:
    try:
        await fn()
    except Exception as exc:  # keep the loop alive whatever a subsystem does
        print(f"   · {label}: skipped ({type(exc).__name__}: {exc})")


async def one_pass(emb: BagOfWordsEmbeddings, llm: BaseLLM) -> None:
    from synapsekit.retrieval.vectorstore import InMemoryVectorStore

    store = InMemoryVectorStore(embedding_backend=emb)

    async def loader() -> None:
        from synapsekit.loaders import TextLoader

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "policy.txt"
            p.write_text(KNOWLEDGE_BASE[0])
            await asyncio.to_thread(TextLoader(str(p)).load)

    async def index() -> None:  # embeddings + vector store
        await store.add(KNOWLEDGE_BASE)

    async def tool() -> None:
        await SearchDocsTool().run(operation="query")

    async def mcp_tool() -> None:
        from synapsekit.mcp.client import MCPToolAdapter

        class _FakeMcpTool:  # matches the MCP tool shape MCPToolAdapter reads
            name = "mcp:web_search"
            description = "MCP web search"
            inputSchema: dict = {}  # noqa: N815 (MCP protocol field name)

        await MCPToolAdapter(_FakeMcpTool()).run(q="refund")

    async def memory() -> None:
        from synapsekit.memory.agent_memory import AgentMemory

        mem = AgentMemory(backend="memory")
        await mem.store(agent_id="support", content="customer asked about a refund")
        await mem.recall(agent_id="support", query="refund")

    async def graph() -> None:
        from synapsekit.retrieval.world_model import WorldModelRAG

        wm = WorldModelRAG(
            vector_store=InMemoryVectorStore(embedding_backend=emb),
            graph_backend="in_memory",
            llm=FakeLLM(LLMConfig(provider="fake", model="fake-1", api_key="")),
        )
        await wm.ingest(["Acme refunded order 48213 on 2026-08-01."])
        await wm.query("refund")

    async def rag_answer() -> None:  # the capstone: real retrieval → real Claude
        hits = await store.search(QUESTION, top_k=2)
        context = "\n".join(_hit_text(h) for h in hits)
        prompt = (
            "Answer the customer using ONLY this policy context.\n\n"
            f"Context:\n{context}\n\nQuestion: {QUESTION}\nAnswer:"
        )
        answer = await llm.generate(prompt)
        print(f"RAG → {answer.strip()}")

    for label, fn in [
        ("loader", loader),
        ("index", index),
        ("tool", tool),
        ("mcp", mcp_tool),
        ("memory", memory),
        ("graph", graph),
        ("rag", rag_answer),
    ]:
        await _guard(label, fn)


async def main() -> None:
    enable(open_browser=True)
    _patch(BagOfWordsEmbeddings, "embed", "embeddings.embed", lambda self, a, k: {"dim": _DIM})
    emb, llm = BagOfWordsEmbeddings(), _make_llm()
    mode = "real Claude" if os.environ.get("ANTHROPIC_API_KEY") else "local FakeLLM"
    print(f"Running every subsystem on a loop ({mode})… open the tab. Ctrl+C to stop.")
    while True:
        await one_pass(emb, llm)
        await asyncio.sleep(1.5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
