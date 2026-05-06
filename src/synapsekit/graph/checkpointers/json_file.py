from __future__ import annotations

import os
from typing import Any

from ..._json import dumps as _json_dumps
from ..._json import loads as _json_loads
from .base import BaseCheckpointer


class JSONFileCheckpointer(BaseCheckpointer):
    """File-based checkpointer using JSON files.

    Each graph checkpoint is stored as a separate JSON file named
    ``{graph_id}.json`` in the given directory.
    """

    def __init__(self, directory: str = ".") -> None:
        self._directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path_for(self, graph_id: str) -> str:
        return os.path.join(self._directory, f"{graph_id}.json")

    def save(self, graph_id: str, step: int, state: dict[str, Any]) -> None:
        path = self._path_for(graph_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_json_dumps({"step": step, "state": state}))

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        path = self._path_for(graph_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = _json_loads(f.read())
        return data["step"], data["state"]

    def delete(self, graph_id: str) -> None:
        path = self._path_for(graph_id)
        if os.path.exists(path):
            os.remove(path)
