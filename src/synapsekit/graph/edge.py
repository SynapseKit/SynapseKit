from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

# A condition function takes the current state and returns a node name (or END).
ConditionFn = Callable[[dict[str, Any]], Union[str, Awaitable[str]]]


@dataclass
class Edge:
    src: str
    dst: str


@dataclass
class ConditionalEdge:
    src: str
    condition_fn: ConditionFn
    # Maps condition_fn return value → destination node name (or END).
    mapping: dict[str, str] = field(default_factory=dict)
