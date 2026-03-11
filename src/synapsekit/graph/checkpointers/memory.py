from __future__ import annotations

import copy
from typing import Any

from .base import BaseCheckpointer


class InMemoryCheckpointer(BaseCheckpointer):
    """Dict-backed in-memory checkpointer with deepcopy semantics."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[int, dict[str, Any]]] = {}

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        self._store[graph_id] = (step, copy.deepcopy(state))

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        entry = self._store.get(graph_id)
        if entry is None:
            return None
        step, state = entry
        return step, copy.deepcopy(state)

    def delete(self, graph_id: str) -> None:
        self._store.pop(graph_id, None)
