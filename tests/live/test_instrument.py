"""Auto-instrumentation of tools / memory (DB) / graphs so they stream to Live.

Real objects only — a real BaseTool subclass and a real in-memory AgentMemory.
No server needed: instrument the classes and toggle the bus directly.
"""

from __future__ import annotations

import inspect

import pytest

from synapsekit.agents.base import BaseTool, ToolResult
from synapsekit.live import bus
from synapsekit.live.instrument import instrument_all


class _CalcTool(BaseTool):
    name = "calc"
    description = "adds"
    parameters = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        return ToolResult(output="42")


class _BoomTool(BaseTool):
    name = "boom"
    description = "raises"
    parameters = {"type": "object", "properties": {}}

    async def run(self, **kwargs: object) -> ToolResult:
        raise RuntimeError("kaboom")


@pytest.fixture(autouse=True)
def _instrumented_bus():
    instrument_all()  # idempotent — patches BaseTool + memory etc. once
    was = bus.enabled
    bus.enabled = True
    bus.clear()
    yield
    bus.enabled = was


def test_run_stays_a_coroutine_after_instrumentation() -> None:
    assert inspect.iscoroutinefunction(BaseTool.run)
    assert inspect.iscoroutinefunction(_CalcTool.run)


async def test_tool_run_publishes_event() -> None:
    await _CalcTool().run(operation="add")
    events = [e for e in bus.history() if e["kind"] == "tool.call"]
    assert events, "tool.call not published"
    assert events[-1]["attributes"]["tool"] == "calc"
    assert events[-1]["attributes"]["operation"] == "add"
    assert events[-1]["status"] == "ok"
    assert events[-1]["duration_ms"] >= 0


async def test_tool_error_is_reported_and_reraised() -> None:
    with pytest.raises(RuntimeError, match="kaboom"):
        await _BoomTool().run()
    events = [e for e in bus.history() if e["kind"] == "tool.call"]
    assert events[-1]["status"] == "error"


async def test_future_subclass_is_instrumented() -> None:
    # A tool class defined *after* instrument_all() must still be covered.
    class _LateTool(BaseTool):
        name = "late"
        description = "x"
        parameters = {"type": "object", "properties": {}}

        async def run(self, **kwargs: object) -> ToolResult:
            return ToolResult(output="ok")

    assert inspect.iscoroutinefunction(_LateTool.run)
    await _LateTool().run()
    assert any(
        e["kind"] == "tool.call" and e["attributes"].get("tool") == "late" for e in bus.history()
    )


async def test_memory_read_and_write_publish() -> None:
    from synapsekit.memory.agent_memory import AgentMemory

    mem = AgentMemory(backend="memory")
    await mem.store(agent_id="a1", content="user prefers dark mode")
    await mem.recall(agent_id="a1", query="mode")
    kinds = [e["kind"] for e in bus.history()]
    assert "memory.write" in kinds
    assert "memory.read" in kinds


async def test_noop_when_bus_disabled() -> None:
    bus.enabled = False
    bus.clear()
    await _CalcTool().run()
    assert bus.history() == []  # nothing published when Live is off


def test_budget_and_audit_publish() -> None:
    from synapsekit.observability.audit_log import AuditLog
    from synapsekit.observability.budget_guard import BudgetGuard, BudgetLimit

    BudgetGuard(BudgetLimit(daily=5.0)).record_spend(0.0043)
    AuditLog().record(model="claude-haiku-4-5", input_text="hi", output_text="hello", user="u1")
    kinds = [e["kind"] for e in bus.history()]
    assert "budget" in kinds and "audit" in kinds
    budget = [e for e in bus.history() if e["kind"] == "budget"][-1]
    assert budget["attributes"]["limit"] == 5.0
    audit = [e for e in bus.history() if e["kind"] == "audit"][-1]
    assert audit["attributes"]["model"] == "claude-haiku-4-5"


def test_sync_loader_publishes(tmp_path) -> None:
    # Loaders are sync and share no base class — the sync wrapper covers them.
    from synapsekit.loaders import TextLoader

    p = tmp_path / "doc.txt"
    p.write_text("hello world")
    TextLoader(str(p)).load()
    events = [e for e in bus.history() if e["kind"] == "loader.load"]
    assert events, "loader.load not published"
    assert events[-1]["attributes"]["loader"] == "TextLoader"
