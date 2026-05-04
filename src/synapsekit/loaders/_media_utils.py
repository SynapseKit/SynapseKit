"""Shared helpers for media loader locator formatting."""

from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    total_seconds = max(0, int(value))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_locator(start: Any, end: Any = None) -> str | None:
    start_ts = _to_float(start)
    end_ts = _to_float(end)
    if start_ts is None and end_ts is None:
        return None
    if start_ts is None:
        return format_seconds(end_ts)
    if end_ts is None or end_ts == start_ts:
        return format_seconds(start_ts)
    return f"{format_seconds(start_ts)}-{format_seconds(end_ts)}"
