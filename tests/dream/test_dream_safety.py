"""Dream Mode resource and mesh-safety tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from synapsekit.audit import AuditTracer, EventKind
from synapsekit.dream import DreamConfig, DreamMode, PowerStatus


class CountingBackend:
    calls = 0

    async def generate(self, prompt: str, **kwargs: object) -> str:
        self.calls += 1
        return "[]"


class FakeMesh:
    def __init__(self) -> None:
        self.reindexed = False

    async def reindex(self) -> None:
        self.reindexed = True

    def duplicates(self, *, limit: int) -> list[object]:
        assert limit == 20
        return [SimpleNamespace(left_path="a.md", right_path="b.md", score=0.91)]


def test_budget_guard_uses_deterministic_path_without_model_call(tmp_path: Path) -> None:
    tracer = AuditTracer(run_id="budget-day")
    tracer.record(EventKind.ERROR, {"message": "failed " + "x" * 10_000})
    backend = CountingBackend()
    mode = DreamMode(
        config=DreamConfig(
            schedule="02:00",
            budget_tokens=1,
            state_path=tmp_path / "state.sqlite3",
            audit_dir=tmp_path / "audit",
        ),
        backend=backend,
    )
    try:
        mode.ingest_traces(tracer.records)
        result = asyncio.run(
            mode.run_once(
                force=True,
                power=PowerStatus(plugged_in=True),
            )
        )
        assert result.lessons
        assert backend.calls == 0
    finally:
        mode.close()


def test_mesh_refresh_reports_candidates_without_merging(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    mode = DreamMode(
        config=DreamConfig(
            schedule="02:00",
            tasks=("consolidate_entities",),
            state_path=tmp_path / "state.sqlite3",
            audit_dir=tmp_path / "audit",
        ),
        mesh=FakeMesh(),  # type: ignore[arg-type]
    )
    try:
        result = asyncio.run(
            mode.run_once(
                now=now,
                force=True,
                power=PowerStatus(plugged_in=True),
            )
        )
        assert result.mesh_reindexed
        assert result.mesh_consolidations[0].left_path == "a.md"
        assert result.mesh_consolidations[0].score == 0.91
    finally:
        mode.close()
