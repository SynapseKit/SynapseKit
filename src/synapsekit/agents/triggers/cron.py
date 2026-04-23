"""Schedule-based agent execution via cron expressions or interval shorthand."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from ...observability.audit_log import AuditLog

_INTERVAL_PATTERN = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")
MissedRunPolicy = Literal["skip", "catch_up"]


@dataclass(frozen=True)
class TriggerResult:
    """Result metadata for a single scheduled trigger execution."""

    scheduled_for: datetime
    started_at: datetime
    finished_at: datetime
    input_text: str
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CronTrigger:
    """Run an agent on a cron schedule or fixed interval."""

    def __init__(
        self,
        agent: Any,
        *,
        prompt: str,
        schedule: str | None = None,
        every: str | None = None,
        timezone: str = "UTC",
        missed_run_policy: MissedRunPolicy = "skip",
        audit_log: AuditLog | None = None,
        result_sink: Callable[[TriggerResult], Awaitable[None] | None] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep_func: Callable[[float], Coroutine[Any, Any, None]] | None = None,
        max_catch_up_runs: int = 100,
    ) -> None:
        if (schedule is None) == (every is None):
            raise ValueError("Provide exactly one of 'schedule' or 'every'.")
        if missed_run_policy not in {"skip", "catch_up"}:
            raise ValueError("missed_run_policy must be 'skip' or 'catch_up'.")
        if max_catch_up_runs < 1:
            raise ValueError("max_catch_up_runs must be >= 1.")

        self.agent = agent
        self.prompt = prompt
        self.schedule = schedule
        self.every = every
        self.timezone = timezone
        self.missed_run_policy = missed_run_policy
        self.audit_log = audit_log
        self.result_sink = result_sink
        self.max_catch_up_runs = max_catch_up_runs

        self._zone = ZoneInfo(timezone)
        self._clock = clock or (lambda: datetime.now(self._zone))
        self._sleep = sleep_func or asyncio.sleep
        self._interval = self._parse_interval(every) if every is not None else None
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._sleep_task: asyncio.Task[None] | None = None
        self._next_run_at: datetime | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def next_run_at(self) -> datetime | None:
        return self._next_run_at

    async def start(self) -> None:
        if self.is_running:
            return
        self._stop_event = asyncio.Event()
        self._next_run_at = self._first_run_after(self._now())
        self._task = asyncio.create_task(
            self._run_loop(), name=f"CronTrigger[{self._source_name()}]"
        )

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return

        self._stop_event.set()
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()

        if asyncio.current_task() is task:
            return

        await task

    async def _run_loop(self) -> None:
        try:
            if self._next_run_at is None:
                self._next_run_at = self._first_run_after(self._now())

            while not self._stop_event.is_set():
                next_run = self._next_run_at
                if next_run is None:
                    next_run = self._first_run_after(self._now())
                    self._next_run_at = next_run

                now = self._now()
                if now < next_run:
                    await self._sleep_until(next_run)
                    continue

                if self.missed_run_policy == "catch_up":
                    scheduled_runs, next_run_at = self._collect_catch_up_runs(next_run)
                else:
                    scheduled_runs = [next_run]
                    next_run_at = None

                for scheduled_for in scheduled_runs:
                    await self._execute_run(scheduled_for)
                    if self._stop_event.is_set():
                        break

                if self._stop_event.is_set():
                    break

                if self.missed_run_policy == "catch_up":
                    self._next_run_at = next_run_at
                else:
                    self._next_run_at = self._next_future_run(scheduled_runs[-1])
        finally:
            self._sleep_task = None
            self._task = None

    async def _sleep_until(self, target: datetime) -> None:
        delay = max(0.0, (target - self._now()).total_seconds())
        self._sleep_task = asyncio.create_task(self._sleep(delay))
        try:
            await self._sleep_task
        except asyncio.CancelledError:
            if not self._stop_event.is_set():
                raise
        finally:
            self._sleep_task = None

    def _collect_catch_up_runs(self, first_due: datetime) -> tuple[list[datetime], datetime]:
        scheduled_runs = [first_due]
        next_run = self._advance(first_due)
        now = self._now()

        while next_run <= now and len(scheduled_runs) < self.max_catch_up_runs:
            scheduled_runs.append(next_run)
            next_run = self._advance(next_run)

        return scheduled_runs, next_run

    def _next_future_run(self, previous_run: datetime) -> datetime:
        next_run = self._advance(previous_run)
        now = self._now()
        while next_run <= now:
            next_run = self._advance(next_run)
        return next_run

    async def _execute_run(self, scheduled_for: datetime) -> None:
        started_at = self._now()
        output: Any = None
        error_text: str | None = None

        try:
            output = await self._invoke_agent(self.prompt)
        except Exception as exc:  # pragma: no cover - behavior preserved via logging/result sink
            error_text = f"{type(exc).__name__}: {exc}"

        finished_at = self._now()
        result = TriggerResult(
            scheduled_for=scheduled_for,
            started_at=started_at,
            finished_at=finished_at,
            input_text=self.prompt,
            output=output,
            error=error_text,
            metadata={
                "timezone": self.timezone,
                "schedule": self.schedule,
                "every": self.every,
                "missed_run_policy": self.missed_run_policy,
            },
        )

        if self.audit_log is not None:
            self.audit_log.record(
                model=self._model_name(),
                input_text=self.prompt,
                output_text="" if output is None else str(output),
                latency_ms=(finished_at - started_at).total_seconds() * 1000,
                user=self._source_name(),
                metadata={
                    **result.metadata,
                    "scheduled_for": scheduled_for.isoformat(),
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "error": error_text,
                },
            )

        if self.result_sink is not None:
            maybe_awaitable = self.result_sink(result)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable

    async def _invoke_agent(self, payload: str) -> Any:
        runner = getattr(self.agent, "run", None)
        if runner is None:
            if not callable(self.agent):
                raise TypeError("agent must be callable or define a 'run' method")
            runner = self.agent

        result = runner(payload)
        if inspect.isawaitable(result):
            return await result
        return result

    def _first_run_after(self, base: datetime) -> datetime:
        return self._advance(base)

    def _advance(self, base: datetime) -> datetime:
        if self._interval is not None:
            return base + self._interval
        return self._cron_next(base)

    def _cron_next(self, base: datetime) -> datetime:
        try:
            from croniter import croniter  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without optional dep
            raise ImportError(
                "CronTrigger schedule support requires the optional dependency 'croniter'. "
                'Install it with: pip install "synapsekit[cron]" (or pip install croniter)'
            ) from exc

        cron = croniter(self.schedule, base)
        next_run = cron.get_next(datetime)
        return self._normalize_datetime(next_run)

    def _now(self) -> datetime:
        current = self._clock()
        return self._normalize_datetime(current)

    def _normalize_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=self._zone)
        return value.astimezone(self._zone)

    def _model_name(self) -> str:
        llm = getattr(getattr(self.agent, "config", None), "llm", None)
        for attr in ("model", "model_name", "name"):
            candidate = getattr(llm, attr, None)
            if candidate:
                return str(candidate)
        return type(self.agent).__name__

    def _source_name(self) -> str:
        agent_id = getattr(getattr(self.agent, "config", None), "agent_id", None)
        if agent_id:
            return str(agent_id)
        return type(self.agent).__name__

    @staticmethod
    def _parse_interval(raw: str | None) -> timedelta:
        if raw is None:
            raise ValueError("every must not be None")

        match = _INTERVAL_PATTERN.fullmatch(raw.strip())
        if match is None:
            raise ValueError("every must use interval shorthand like '30m', '1h', '10s', or '2d'.")

        value = int(match.group("value"))
        unit = match.group("unit")
        unit_map = {
            "s": "seconds",
            "m": "minutes",
            "h": "hours",
            "d": "days",
        }
        return timedelta(**{unit_map[unit]: value})
