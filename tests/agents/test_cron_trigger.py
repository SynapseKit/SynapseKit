from __future__ import annotations

import asyncio
import builtins
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import synapsekit.agents.triggers.cron as cron_module
from synapsekit import CronTrigger as TopLevelCronTrigger
from synapsekit import TriggerResult as TopLevelTriggerResult
from synapsekit.agents.triggers import CronTrigger, TriggerResult
from synapsekit.observability import AuditLog


@dataclass
class ControlledClock:
    current: datetime
    advances: list[float] | None = None
    block_after_calls: int | None = None

    def __post_init__(self) -> None:
        self.sleep_calls: list[float] = []
        self._blocker = asyncio.Event()

    def now(self) -> datetime:
        return self.current

    async def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        index = len(self.sleep_calls) - 1
        actual = seconds
        if self.advances is not None and index < len(self.advances):
            actual = self.advances[index]
        self.current += timedelta(seconds=actual)
        await asyncio.sleep(0)
        if self.block_after_calls is not None and len(self.sleep_calls) >= self.block_after_calls:
            await self._blocker.wait()


async def spin(count: int = 10) -> None:
    for _ in range(count):
        await asyncio.sleep(0)


def test_trigger_exports_are_importable() -> None:
    assert TopLevelCronTrigger is CronTrigger
    assert TopLevelTriggerResult is TriggerResult


@pytest.mark.asyncio
async def test_interval_trigger_runs_and_logs_to_audit_log() -> None:
    clock = ControlledClock(
        current=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        block_after_calls=2,
    )
    agent = SimpleNamespace(
        run=AsyncMock(return_value="summary ready"),
        config=SimpleNamespace(agent_id="daily-brief"),
    )
    audit_log = AuditLog(backend="memory")

    trigger = CronTrigger(
        agent=agent,
        every="5s",
        input="summarize yesterday",
        timezone="UTC",
        audit_log=audit_log,
        clock=clock.now,
        sleep_func=clock.sleep,
    )

    await trigger.start()
    await spin()

    assert agent.run.await_count == 1
    assert len(audit_log) == 1
    entry = audit_log.query(limit=1)[0]
    assert entry.input_text == "summarize yesterday"
    assert entry.output_text == "summary ready"
    assert entry.user == "daily-brief"

    await trigger.stop()


@pytest.mark.asyncio
async def test_cron_expression_uses_croniter_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCronIter:
        def __init__(self, expr: str, base: datetime) -> None:
            assert expr == "*/15 * * * *"
            self.expr = expr
            self.base = base

        def get_next(self, result_type: type[datetime]) -> datetime:
            assert result_type is datetime
            self.base = self.base + timedelta(minutes=15)
            return self.base

    monkeypatch.setitem(sys.modules, "croniter", SimpleNamespace(croniter=FakeCronIter))

    clock = ControlledClock(
        current=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        block_after_calls=2,
    )
    agent = SimpleNamespace(run=AsyncMock(return_value="ran"))

    trigger = CronTrigger(
        agent=agent,
        schedule="*/15 * * * *",
        input="tick",
        timezone="UTC",
        clock=clock.now,
        sleep_func=clock.sleep,
    )

    await trigger.start()
    await spin()

    assert clock.sleep_calls[0] == 900
    assert agent.run.await_count == 1

    await trigger.stop()


@pytest.mark.asyncio
async def test_catch_up_replays_missed_interval_runs() -> None:
    clock = ControlledClock(
        current=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        advances=[16, 5],
        block_after_calls=2,
    )
    agent = SimpleNamespace(run=AsyncMock(return_value="ok"))

    trigger = CronTrigger(
        agent=agent,
        every="5s",
        input="check queue",
        timezone="UTC",
        missed_run_policy="catch_up",
        clock=clock.now,
        sleep_func=clock.sleep,
    )

    await trigger.start()
    await spin()

    assert agent.run.await_count == 3

    await trigger.stop()


@pytest.mark.asyncio
async def test_stop_waits_for_in_flight_run_to_finish() -> None:
    clock = ControlledClock(current=datetime(2026, 1, 1, 9, 0, tzinfo=UTC))
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def run(_: str) -> str:
        started.set()
        await release.wait()
        finished.set()
        return "done"

    agent = SimpleNamespace(run=run)
    trigger = CronTrigger(
        agent=agent,
        every="5s",
        input="do work",
        timezone="UTC",
        clock=clock.now,
        sleep_func=clock.sleep,
    )

    await trigger.start()
    await asyncio.wait_for(started.wait(), timeout=1)

    stop_task = asyncio.create_task(trigger.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()

    release.set()
    await asyncio.wait_for(stop_task, timeout=1)

    assert finished.is_set()


@pytest.mark.asyncio
async def test_skip_policy_drops_overdue_runs_after_long_execution() -> None:
    clock = ControlledClock(
        current=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        block_after_calls=2,
    )
    audit_log = AuditLog(backend="memory")

    async def run(_: str) -> str:
        if len(audit_log) == 0:
            clock.current += timedelta(seconds=16)
        return "ok"

    trigger = CronTrigger(
        agent=SimpleNamespace(run=run),
        every="5s",
        input="process once",
        timezone="UTC",
        audit_log=audit_log,
        clock=clock.now,
        sleep_func=clock.sleep,
    )

    await trigger.start()
    await spin()
    await trigger.stop()

    assert len(audit_log) == 1
    assert audit_log.query(limit=1)[0].metadata["scheduled_for"] == "2026-01-01T09:00:05+00:00"
    assert trigger.next_run_at == datetime(2026, 1, 1, 9, 0, 25, tzinfo=UTC)


@pytest.mark.asyncio
async def test_result_sink_can_stop_trigger_without_self_await_deadlock() -> None:
    clock = ControlledClock(current=datetime(2026, 1, 1, 9, 0, tzinfo=UTC))
    stopped = asyncio.Event()
    trigger: CronTrigger | None = None

    async def sink(_: object) -> None:
        assert trigger is not None
        await trigger.stop()
        stopped.set()

    trigger = CronTrigger(
        agent=SimpleNamespace(run=AsyncMock(return_value="done")),
        every="5s",
        input="stop after first run",
        timezone="UTC",
        clock=clock.now,
        sleep_func=clock.sleep,
        result_sink=sink,
    )

    await trigger.start()
    await asyncio.wait_for(stopped.wait(), timeout=1)

    assert not trigger.is_running


def test_default_clock_is_timezone_aware(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDateTime:
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            assert tz is not None
            return datetime(2026, 1, 1, 9, 0, tzinfo=tz)

    monkeypatch.setattr(cron_module, "datetime", FakeDateTime)
    trigger = CronTrigger(
        agent=SimpleNamespace(run=AsyncMock(return_value="done")),
        every="5s",
        input="aware now",
        timezone="Asia/Kolkata",
    )

    assert trigger._now() == datetime(
        2026,
        1,
        1,
        9,
        0,
        tzinfo=trigger.next_run_at.tzinfo if trigger.next_run_at else trigger._zone,
    )


@pytest.mark.asyncio
async def test_missing_croniter_error_mentions_synapsekit_cron_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "croniter":
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("croniter", None)

    trigger = CronTrigger(
        agent=SimpleNamespace(run=AsyncMock(return_value="done")),
        schedule="0 9 * * *",
        input="cron",
        timezone="UTC",
        clock=lambda: datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
        sleep_func=AsyncMock(),
    )

    with pytest.raises(ImportError, match="synapsekit\\[cron\\]"):
        await trigger.start()
