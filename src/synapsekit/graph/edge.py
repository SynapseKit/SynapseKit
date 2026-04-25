from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

# A condition function takes the current state and returns a node name (or END).
ConditionFn = Callable[[dict[str, Any]], str | Awaitable[str]]


@dataclass
class Edge:
    src: str
    dst: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionalEdge:
    src: str
    condition_fn: ConditionFn
    # Maps condition_fn return value → destination node name (or END).
    mapping: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
