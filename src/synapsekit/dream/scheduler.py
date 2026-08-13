"""Schedule, idle, and power gates for Dream Mode.

The scheduler is deliberately small and injectable.  The default monitors use
only the standard library, while tests and desktop integrations can provide
their own power and idle readings without mocking operating-system calls.
"""

from __future__ import annotations

import asyncio
import ctypes
import sys
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Protocol, cast

from .types import PowerStatus


class PowerMonitor(Protocol):
    """Provider for the current power state."""

    def status(self) -> PowerStatus: ...


class IdleMonitor(Protocol):
    """Provider for seconds since the last user input."""

    def idle_seconds(self) -> float: ...


@dataclass(frozen=True)
class DreamSchedule:
    """Parsed ``idle_30m or 02:00`` schedule expression."""

    idle_after_seconds: float | None = None
    clock_times: tuple[time, ...] = ()

    @classmethod
    def parse(cls, expression: str, *, default_idle_seconds: float = 1800.0) -> DreamSchedule:
        if not expression.strip():
            raise ValueError("Dream Mode schedule must not be empty")
        idle_after: float | None = None
        clocks: list[time] = []
        for raw_part in expression.split(" or "):
            part = raw_part.strip().lower()
            if part == "idle":
                idle_after = default_idle_seconds
                continue
            if part.startswith("idle_") and part.endswith("m"):
                try:
                    minutes = float(part[5:-1])
                except ValueError as exc:
                    raise ValueError(f"invalid idle schedule component: {raw_part!r}") from exc
                if minutes < 0:
                    raise ValueError("idle duration cannot be negative")
                idle_after = minutes * 60
                continue
            if part.startswith("idle_") and part.endswith("s"):
                try:
                    seconds = float(part[5:-1])
                except ValueError as exc:
                    raise ValueError(f"invalid idle schedule component: {raw_part!r}") from exc
                if seconds < 0:
                    raise ValueError("idle duration cannot be negative")
                idle_after = seconds
                continue
            try:
                hour, minute = (int(value) for value in part.split(":", 1))
                clocks.append(time(hour=hour, minute=minute))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"invalid Dream Mode schedule component {raw_part!r}; expected idle_30m or HH:MM"
                ) from exc
        if idle_after is None and not clocks:
            raise ValueError("Dream Mode schedule contains no usable trigger")
        return cls(idle_after_seconds=idle_after, clock_times=tuple(clocks))

    def due(self, now: datetime, *, idle_seconds: float) -> bool:
        """Return whether a trigger is active at ``now``."""

        local_now = now.astimezone()
        idle_due = self.idle_after_seconds is not None and idle_seconds >= self.idle_after_seconds
        clock_due = any(
            local_now.hour == clock.hour and local_now.minute == clock.minute
            for clock in self.clock_times
        )
        return idle_due or clock_due

    def trigger_key(self, now: datetime, *, idle_seconds: float) -> str | None:
        """Return a stable per-day key so a poll loop runs once per trigger."""

        local_now = now.astimezone()
        if self.clock_times and any(
            local_now.hour == clock.hour and local_now.minute == clock.minute
            for clock in self.clock_times
        ):
            return (
                f"clock:{local_now.date().isoformat()}:{local_now.hour:02d}:{local_now.minute:02d}"
            )
        if self.idle_after_seconds is not None and idle_seconds >= self.idle_after_seconds:
            return f"idle:{local_now.date().isoformat()}"
        return None


class SystemPowerMonitor:
    """Best-effort standard-library power monitor for Windows and Linux."""

    def status(self) -> PowerStatus:
        if sys.platform == "win32":
            return self._windows_status()
        if sys.platform.startswith("linux"):
            return self._linux_status()
        return PowerStatus(plugged_in=False, known=False)

    @staticmethod
    def _windows_status() -> PowerStatus:
        class _SystemPowerStatus(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("Reserved", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        value = _SystemPowerStatus()
        try:
            ok = cast(Any, ctypes.windll).kernel32.GetSystemPowerStatus(ctypes.byref(value))
        except (AttributeError, OSError):
            return PowerStatus(plugged_in=False, known=False)
        if not ok or value.ACLineStatus == 255:
            return PowerStatus(plugged_in=False, known=False)
        percent = None if value.BatteryLifePercent == 255 else int(value.BatteryLifePercent)
        return PowerStatus(plugged_in=value.ACLineStatus == 1, battery_percent=percent)

    @staticmethod
    def _linux_status() -> PowerStatus:
        power_root = Path("/sys/class/power_supply")
        try:
            online_values = [
                (entry / "online").read_text(encoding="ascii").strip()
                for entry in power_root.glob("AC*/online")
            ]
            capacity_values = [
                int((entry / "capacity").read_text(encoding="ascii").strip())
                for entry in power_root.glob("BAT*/capacity")
            ]
        except (OSError, ValueError):
            return PowerStatus(plugged_in=False, known=False)
        if not online_values:
            return PowerStatus(plugged_in=False, known=False)
        return PowerStatus(
            plugged_in=any(value == "1" for value in online_values),
            battery_percent=capacity_values[0] if capacity_values else None,
        )


class SystemIdleMonitor:
    """Return system idle seconds where the platform exposes that signal."""

    def idle_seconds(self) -> float:
        if sys.platform != "win32":
            return 0.0

        class _LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = _LastInputInfo(ctypes.sizeof(_LastInputInfo), 0)
        try:
            if not cast(Any, ctypes.windll).user32.GetLastInputInfo(ctypes.byref(info)):
                return 0.0
            tick = cast(Any, ctypes.windll).kernel32.GetTickCount()
        except (AttributeError, OSError):
            return 0.0
        elapsed_ms = (int(tick) - int(info.dwTime)) & 0xFFFFFFFF
        return elapsed_ms / 1000.0


@dataclass
class DreamScheduler:
    """Stateful trigger gate used by ``DreamMode.run_forever``."""

    schedule: DreamSchedule
    last_trigger_key: str | None = None

    def should_run(
        self,
        now: datetime,
        *,
        idle_seconds: float,
        power: PowerStatus,
        require_plugged_in: bool,
    ) -> tuple[bool, str]:
        if require_plugged_in and (not power.known or not power.plugged_in):
            return False, "Dream Mode requires a known plugged-in power source"
        key = self.schedule.trigger_key(now, idle_seconds=idle_seconds)
        if key is None or not self.schedule.due(now, idle_seconds=idle_seconds):
            return False, "schedule is not due"
        if key == self.last_trigger_key:
            return False, "schedule trigger already consumed"
        self.last_trigger_key = key
        return True, key


async def wait_for_stop(stop_event: asyncio.Event, seconds: float) -> None:
    """Sleep while remaining cancellable for scheduler shutdown."""

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return
