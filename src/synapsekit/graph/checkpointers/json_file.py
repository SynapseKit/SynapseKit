from __future__ import annotations

import json
import os
from typing import Any

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
            json.dump({"step": step, "state": state}, f)

    def load(self, graph_id: str) -> tuple[int, dict[str, Any]] | None:
        path = self._path_for(graph_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["step"], data["state"]

    def delete(self, graph_id: str) -> None:
        path = self._path_for(graph_id)
        if os.path.exists(path):
            os.remove(path)
