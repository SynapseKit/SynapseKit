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
