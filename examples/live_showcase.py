"""SynapseKit Live — the full showcase, paced for a screen recording.

    export ANTHROPIC_API_KEY=sk-ant-...   # optional: real Claude, else a local FakeLLM
    python examples/live_showcase.py

Opens http://127.0.0.1:7900 and walks through the whole framework as a series of
labelled runs (use the "Runs" dropdown to browse history). Every UI feature
lights up: the subsystem activity strip, Activity/Logs/Errors tabs, click-to-
expand prompt+response, cost/token meters, latency sparkline, budget gauge,
signed audit, the flame graph (nested spans), the knowledge-graph canvas (real
entities), and a human-in-the-loop Approve/Deny gate. Paced with pauses so it
records cleanly. Ctrl+C to stop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import AsyncGenerator

import numpy as np

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.embeddings.backend import SynapsekitEmbeddings
from synapsekit.live import enable, new_run, request_approval
from synapsekit.live.instrument import _patch
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.observe.runtime import end_span, start_span

log = logging.getLogger("support-agent")
PACE = float(os.environ.get("SHOWCASE_PACE", "1.1"))
_DIM = 48
KB = [
    "Refunds are processed within 5 business days to the original payment method.",
    "You can return any item within 30 days of delivery for a full refund.",
    "Gift cards are non-refundable and cannot be exchanged for cash.",
    "Orders over $50 ship free; standard shipping takes 3 to 5 days.",
]


class BoW(SynapsekitEmbeddings):
    async def embed(self, texts: list[str]) -> np.ndarray:
        v = np.zeros((len(texts), _DIM), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[i, hash(w) % _DIM] += 1.0
            v[i] /= np.linalg.norm(v[i]) or 1.0
        return v


class FakeLLM(BaseLLM):
    async def stream(self, prompt: str, **kw: object) -> AsyncGenerator[str, None]:
        for w in ("Refunds", "are", "processed", "within", "5", "business", "days."):
            yield w + " "


class LookupOrderTool(BaseTool):
    name = "lookup_order"
    description = "Look up an order by id."
    parameters = {"type": "object", "properties": {"id": {"type": "string"}}}

    async def run(self, **kwargs: object) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(output="order refunded")


class SendEmailTool(BaseTool):
    name = "send_email"
    description = "Send an email (guarded)."
    parameters = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        raise RuntimeError("SMTP gateway timed out (demo error)")


def _hit(h: object) -> str:
    return str(h.get("text", h)) if isinstance(h, dict) else str(getattr(h, "text", h))


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


async def phase_rag(store, llm) -> None:
    new_run("RAG + Claude")
    log.info("Customer: 'How long do refunds take?'")
    await asyncio.sleep(PACE)
    hits = await store.search("How long do refunds take?", top_k=2)
    ctx = "\n".join(_hit(h) for h in hits)
    try:
        ans = await llm.generate(f"Answer using only:\n{ctx}\nQ: How long do refunds take?\nA:")
        log.info("Answered from policy context (click the 🧠 event to see prompt+response)")
    except Exception as exc:  # bad/expired key etc. — shows up in the Errors tab
        ans = "(LLM unavailable)"
        log.error(f"LLM call failed: {exc}")
    await asyncio.sleep(PACE)
    from synapsekit.observability.audit_log import AuditLog
    from synapsekit.observability.budget_guard import BudgetGuard, BudgetLimit

    BudgetGuard(BudgetLimit(daily=5.0)).record_spend(0.0042)
    AuditLog().record(model="claude-haiku-4-5", input_text="…", output_text=ans, user="demo")
    await asyncio.sleep(PACE)


async def phase_agent() -> None:
    new_run("Agent · tools · MCP · memory")
    from synapsekit.mcp.client import MCPToolAdapter
    from synapsekit.memory.agent_memory import AgentMemory

    async def step(name: str, ms: float) -> None:
        s = start_span(name)
        await asyncio.sleep(ms)
        end_span(s)

    log.info("Agent planning a multi-step response…")
    root = start_span("agent.run")  # nested spans → flame graph
    await step("plan", 0.03)
    await LookupOrderTool().run(id="48213", operation="get")
    await step("reason", 0.05)

    class _Mcp:
        name = "mcp:web_search"
        description = "MCP web search"
        inputSchema: dict = {}  # noqa: N815

    await MCPToolAdapter(_Mcp()).run(q="refund policy")
    mem = AgentMemory(backend="memory")
    await mem.store(agent_id="support", content="customer 48213 asked about a refund")
    await mem.recall(agent_id="support", query="refund")
    await step("respond", 0.04)
    end_span(root)
    log.info("Open the Flame graph below to see the nested agent trace")
    await asyncio.sleep(PACE)


async def phase_graph(emb) -> None:
    new_run("Knowledge graph")
    from synapsekit.retrieval.vectorstore import InMemoryVectorStore
    from synapsekit.retrieval.world_model import WorldModelRAG

    log.info("Ingesting facts into the live knowledge graph…")
    wm = WorldModelRAG(
        vector_store=InMemoryVectorStore(embedding_backend=emb),
        graph_backend="in_memory",
        llm=FakeLLM(LLMConfig(provider="fake", model="fake-1", api_key="")),
    )
    await wm.ingest(
        [
            "Acme Corp refunded Alice for order 48213 on 2026-08-01.",
            "Alice lives in Berlin and works at Acme Corp.",
            "Bob returned a gift card to Acme Corp.",
        ]
    )
    await wm.query("refund")
    log.info("Watch the knowledge-graph canvas — real entities and relations")
    await asyncio.sleep(PACE)


async def phase_hitl() -> None:
    new_run("Human-in-the-loop + error")
    log.warning("Agent wants to send an email — awaiting your approval in the UI")
    approved = await request_approval(
        "send_email", "to customer@acme.com", timeout=45, default=True
    )
    if approved:
        log.info("Approved — sending (will hit a demo error to show the Errors tab)")
        try:
            await SendEmailTool().run()
        except RuntimeError:
            log.error("send_email failed — see the Errors tab + traceback")
    else:
        log.info("Denied — not sending")
    await asyncio.sleep(PACE)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    enable(open_browser=True)
    _patch(BoW, "embed", "embeddings.embed", lambda self, a, k: {"dim": _DIM})
    emb, llm = BoW(), _make_llm()
    from synapsekit.retrieval.vectorstore import InMemoryVectorStore

    store = InMemoryVectorStore(embedding_backend=emb)
    await store.add(KB)
    mode = "real Claude" if os.environ.get("ANTHROPIC_API_KEY") else "local FakeLLM"
    print(f"Showcase running ({mode}) — open the tab. Ctrl+C to stop.")
    phases = [
        ("RAG", lambda: phase_rag(store, llm)),
        ("agent", phase_agent),
        ("graph", lambda: phase_graph(emb)),
        ("hitl", phase_hitl),
    ]
    while True:
        for name, fn in phases:
            try:
                await fn()
            except Exception as exc:  # never let one phase stop the showcase
                log.error(f"{name} phase error: {exc}")
        await asyncio.sleep(PACE * 2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
