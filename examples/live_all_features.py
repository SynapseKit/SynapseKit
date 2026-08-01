"""Watch (almost) EVERY SynapseKit subsystem stream into the Live dashboard.

    python examples/live_all_features.py

Opens http://127.0.0.1:7900 and runs a loop that exercises loaders, embeddings,
vector search / retrieval, tools, an MCP-style tool, agent memory (DB), the
knowledge graph, and an LLM call — all real SynapseKit objects, auto-instrumented
by ``synapsekit.live``. Set ANTHROPIC_API_KEY to use a real Claude call; otherwise
a local FakeLLM stands in. The server stays up as long as this process runs
(Ctrl+C to stop) — that's why you can keep the browser tab open.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.embeddings.backend import SynapsekitEmbeddings
from synapsekit.live import enable
from synapsekit.live.instrument import _patch  # instrument the demo's own classes too
from synapsekit.llm.base import BaseLLM, LLMConfig


class DemoEmbeddings(SynapsekitEmbeddings):
    """Tiny deterministic embeddings so vector/graph run with no model download."""

    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [[(hash(f"{t}:{i}") % 1000) / 1000.0 for i in range(16)] for t in texts],
            dtype="float32",
        )


class FakeLLM(BaseLLM):
    async def stream(self, prompt: str, **kw: object) -> AsyncGenerator[str, None]:
        for word in ("Your", "refund", "is", "on", "the", "way."):
            yield word + " "


class SearchDocsTool(BaseTool):
    name = "search_docs"
    description = "Search the docs."
    parameters = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        return ToolResult(output="found 3 docs")


async def _guard(label: str, fn) -> None:
    """Run one subsystem block; never let a single one break the whole pass."""
    try:
        await fn()
    except Exception as exc:
        print(f"   · {label}: skipped ({type(exc).__name__}: {exc})")


async def one_pass() -> None:
    emb = DemoEmbeddings()
    # instrument the demo's own embeddings subclass so it shows too
    _patch(DemoEmbeddings, "embed", "embeddings.embed", lambda self, a, k: {"dim": 16})

    async def loader() -> None:
        from synapsekit.loaders import TextLoader

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "policy.txt"
            p.write_text("Refunds are processed within 5 business days.")
            await asyncio.to_thread(TextLoader(str(p)).load)

    async def retrieval() -> None:
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(embedding_backend=emb)
        await store.add(["refund policy", "shipping policy", "returns"])
        await store.search("refund", top_k=2)

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
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore
        from synapsekit.retrieval.world_model import WorldModelRAG

        wm = WorldModelRAG(
            vector_store=InMemoryVectorStore(embedding_backend=emb),
            graph_backend="in_memory",
            llm=FakeLLM(LLMConfig(provider="fake", model="fake-1", api_key="")),
        )
        await wm.ingest(["Acme refunded order 48213 on 2026-08-01."])
        await wm.query("refund")

    async def llm() -> None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            from synapsekit.llm.anthropic import AnthropicLLM

            model: BaseLLM = AnthropicLLM(
                LLMConfig(
                    provider="anthropic",
                    model="claude-haiku-4-5-20251001",
                    api_key=key,
                    max_tokens=20,
                )
            )
        else:
            model = FakeLLM(LLMConfig(provider="fake", model="fake-1", api_key=""))
        await model.generate("Reply in one short sentence about the refund.")

    for label, fn in [
        ("loader", loader),
        ("retrieval", retrieval),
        ("tool", tool),
        ("mcp", mcp_tool),
        ("memory", memory),
        ("graph", graph),
        ("llm", llm),
    ]:
        await _guard(label, fn)


async def main() -> None:
    enable(open_browser=True)
    print("Running every subsystem on a loop… open the tab. Ctrl+C to stop.")
    while True:
        await one_pass()
        await asyncio.sleep(1.5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
