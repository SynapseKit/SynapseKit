"""Flame-graph trace events, graph snapshots, and async-concurrency safety."""

from __future__ import annotations

import asyncio

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.live import bus, publish_graph
from synapsekit.live.instrument import instrument_all
from synapsekit.observe.runtime import InMemoryExporter, configure, end_span, start_span


def test_root_span_publishes_flame_tree() -> None:
    configure(exporter=InMemoryExporter())
    bus.enabled = True
    bus.clear()
    root = start_span("agent.run")
    child = start_span("retriever.search")
    end_span(child)
    end_span(root)
    traces = [e for e in bus.history() if e["kind"] == "trace"]
    assert traces, "no flame-graph trace event for a root span with children"
    tree = traces[-1]["attributes"]["tree"]
    assert tree["name"] == "agent.run"
    assert any(c["name"] == "retriever.search" for c in tree["children"])


def test_publish_graph_snapshot() -> None:
    bus.enabled = True
    bus.clear()
    publish_graph([{"id": "Acme", "group": "graph"}, {"id": "refund"}], [["Acme", "refund"]])
    snaps = [e for e in bus.history() if e["kind"] == "graph.snapshot"]
    assert snaps
    assert len(snaps[-1]["attributes"]["nodes"]) == 2
    assert snaps[-1]["attributes"]["edges"] == [["Acme", "refund"]]


class _SleepTool(BaseTool):
    name = "sleeper"
    description = "x"
    parameters = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        await asyncio.sleep(0.01)
        return ToolResult(output="ok")


def test_worldmodel_ingest_snapshots_real_entities() -> None:
    import re
    from collections.abc import AsyncGenerator

    import numpy as np

    from synapsekit.embeddings.backend import SynapsekitEmbeddings
    from synapsekit.llm.base import BaseLLM, LLMConfig
    from synapsekit.retrieval.vectorstore import InMemoryVectorStore
    from synapsekit.retrieval.world_model import WorldModelRAG

    class _Emb(SynapsekitEmbeddings):
        async def embed(self, texts: list[str]) -> np.ndarray:
            v = np.zeros((len(texts), 16), dtype="float32")
            for i, t in enumerate(texts):
                for w in re.findall(r"[a-z]+", t.lower()):
                    v[i, hash(w) % 16] += 1
                v[i] /= np.linalg.norm(v[i]) or 1
            return v

    class _LLM(BaseLLM):
        async def stream(self, prompt: str, **kw: object) -> AsyncGenerator[str, None]:
            yield "ok"

    instrument_all()
    bus.enabled = True
    bus.clear()

    async def go() -> None:
        wm = WorldModelRAG(
            vector_store=InMemoryVectorStore(embedding_backend=_Emb()),
            graph_backend="in_memory",
            llm=_LLM(LLMConfig(provider="fake", model="f", api_key="")),
        )
        await wm.ingest(["Acme Corp refunded Alice for order 48213."])

    asyncio.run(go())
    snaps = [e for e in bus.history() if e["kind"] == "graph.snapshot"]
    assert snaps, "ingest did not snapshot the entity graph"
    labels = [n["label"] for n in snaps[-1]["attributes"]["nodes"]]
    assert any("Acme" in x or "Alice" in x for x in labels)


def test_concurrent_async_calls_all_captured() -> None:
    instrument_all()
    bus.enabled = True
    bus.clear()

    async def blast() -> None:
        await asyncio.gather(*[_SleepTool().run() for _ in range(25)])

    asyncio.run(blast())
    calls = [e for e in bus.history() if e["kind"] == "tool.call"]
    assert len(calls) == 25  # nothing dropped under concurrency
    seqs = [e["seq"] for e in bus.history()]
    assert len(seqs) == len(set(seqs))  # thread/async-safe: unique sequence ids
