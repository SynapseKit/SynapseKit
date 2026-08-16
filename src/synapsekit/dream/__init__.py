"""Local-first overnight self-reflection for SynapseKit agents."""

from .core import DreamMode, load_dream_report, render_briefing
from .scheduler import DreamSchedule, DreamScheduler, SystemIdleMonitor, SystemPowerMonitor
from .store import DreamStateStore
from .types import (
    DEFAULT_TASKS,
    DreamConfig,
    DreamRunResult,
    DreamTask,
    Lesson,
    MeshConsolidation,
    PowerStatus,
    StaleMemory,
    TraceWindow,
)

__all__ = [
    "DEFAULT_TASKS",
    "DreamConfig",
    "DreamMode",
    "DreamRunResult",
    "DreamSchedule",
    "DreamScheduler",
    "DreamStateStore",
    "DreamTask",
    "Lesson",
    "MeshConsolidation",
    "PowerStatus",
    "StaleMemory",
    "SystemIdleMonitor",
    "SystemPowerMonitor",
    "TraceWindow",
    "load_dream_report",
    "render_briefing",
]
