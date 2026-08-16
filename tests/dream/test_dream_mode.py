"""Focused Dream Mode workflow tests."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from synapsekit.audit import AuditTracer, EventKind, Verdict, verify
from synapsekit.dream import DreamConfig, DreamMode, DreamSchedule, PowerStatus


class FakeBackend:
    async def generate(self, prompt: str, **kwargs: object) -> str:
        if "Memory Writer" in prompt or "propose updates" in prompt:
            return (
                '[{"file_path":"MEMORY.md","section":"general",'
                '"fact_key":"mesh_retry_rule",'
                '"proposed_addition":"Retry mesh retrieval after transient failures.",'
                '"rationale":"Repeated retrieval correction.",'
                '"evidence":"retrieval failed; should retry with mesh"}]'
            )
        return "not a lesson response"


def _trace(at: datetime) -> AuditTracer:
    tracer = AuditTracer(run_id="day-1")
    tracer.record(
        EventKind.ERROR,
        {"message": "retrieval failed; should retry with mesh"},
        timestamp=at,
    )
    return tracer


def _multi_record_trace(at: datetime, *, run_id: str = "chain-run") -> AuditTracer:
    """A run whose records all share one timestamp — so a (timestamp,
    event_id) sort scrambles them out of hash-chain order."""

    tracer = AuditTracer(run_id=run_id)
    tracer.record(EventKind.DECISION, {"message": "start retrieval"}, timestamp=at)
    tracer.record(
        EventKind.ERROR,
        {"message": "retrieval failed; should retry with mesh"},
        timestamp=at,
    )
    tracer.record(EventKind.DECISION, {"message": "retry with mesh"}, timestamp=at)
    return tracer


def test_valid_trace_groups_reconstructs_scrambled_chain() -> None:
    # Records arriving out of chain order (the store sorts by
    # (timestamp, event_id); a random event_id scrambles tied timestamps).
    # Reversed order would fail verify_chain if trusted verbatim; the fix
    # relinks by prev_hash/hash before verifying, so nothing is dropped.
    tracer = _multi_record_trace(datetime.now(UTC))
    scrambled = list(reversed(tracer.records))

    kept = DreamMode._valid_trace_groups(scrambled)

    assert len(kept) == len(tracer.records)
    assert {r.event_id for r in kept} == {r.event_id for r in tracer.records}


def test_valid_trace_groups_drops_incomplete_chain() -> None:
    # A run missing an interior link is not a single verifiable chain and
    # must be dropped (rather than silently accepted).
    tracer = _multi_record_trace(datetime.now(UTC))
    broken = [tracer.records[0], tracer.records[2]]  # drop the middle link

    assert DreamMode._valid_trace_groups(broken) == []


def test_schedule_parses_idle_and_clock() -> None:
    schedule = DreamSchedule.parse("idle_30m or 02:00")
    now = datetime.now().astimezone().replace(hour=2, minute=0, second=0, microsecond=0)
    assert schedule.due(now, idle_seconds=0)
    assert schedule.trigger_key(now, idle_seconds=0) == f"clock:{now.strftime('%Y-%m-%d')}:02:00"
    assert schedule.due(now, idle_seconds=1800)


def test_unplugged_run_is_skipped(tmp_path: Path) -> None:
    mode = DreamMode(
        config=DreamConfig(
            schedule="02:00",
            state_path=tmp_path / "state.sqlite3",
            audit_dir=tmp_path / "audit",
        )
    )
    try:
        result = asyncio.run(
            mode.run_once(
                force=True,
                power=PowerStatus(plugged_in=False, battery_percent=50),
            )
        )
        assert result.status == "skipped"
        assert result.audit_path is None
    finally:
        mode.close()


def test_end_to_end_run_proposes_pending_patch_and_signed_audit(tmp_path: Path) -> None:
    memory_path = tmp_path / "MEMORY.md"
    memory_path.write_text("# General\n\nExisting context.\n", encoding="utf-8")
    old_time = (datetime.now(UTC) - timedelta(days=120)).timestamp()
    os.utime(memory_path, (old_time, old_time))

    now = datetime.now(UTC)
    mode = DreamMode(
        config=DreamConfig(
            schedule="02:00",
            lookback_hours=4,
            stale_after_days=90,
            state_path=tmp_path / "state.sqlite3",
            audit_dir=tmp_path / "audit",
        ),
        backend=FakeBackend(),
        memory_paths=[memory_path],
    )
    try:
        assert asyncio.run(mode.ingest_traces(_trace(now - timedelta(hours=1)).records)) == 1
        result = asyncio.run(
            mode.run_once(
                now=now,
                force=True,
                power=PowerStatus(plugged_in=True, battery_percent=100),
            )
        )
        assert result.status == "completed"
        assert result.traces_replayed == 1
        assert result.lessons
        assert result.patch_ids
        assert result.stale_memories
        assert result.audit_path is not None
        assert Path(result.audit_path).exists()
        assert verify(result.audit_path).verdict == Verdict.UNVERIFIABLE
        assert "human review required" in mode.morning_briefing(result)
    finally:
        mode.close()


def test_state_store_last_run_round_trips_typed_report(tmp_path: Path) -> None:
    mode = DreamMode(
        config=DreamConfig(
            schedule="02:00",
            state_path=tmp_path / "state.sqlite3",
            audit_dir=tmp_path / "audit",
        )
    )
    try:
        tracer = _trace(datetime.now(UTC) - timedelta(minutes=10))
        asyncio.run(mode.ingest_traces(tracer.records))
        result = asyncio.run(
            mode.run_once(force=True, power=PowerStatus(plugged_in=True, battery_percent=100))
        )
        loaded = mode.state.last_run()
        assert loaded is not None
        assert loaded.run_id == result.run_id
        assert loaded.status == "completed"
    finally:
        mode.close()
