"""Local and KnowledgeMesh-enriched completion."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any


class CompletionEngine:
    def __init__(self, mesh: Any | None = None) -> None:
        self.mesh = mesh

    async def complete(self, prefix: str, *, cwd: str | Path | None = None) -> list[str]:
        root = Path(cwd or Path.cwd())
        local = await asyncio.to_thread(self._local, prefix, root)
        if self.mesh is None or not prefix.strip():
            return local
        mesh_items = await asyncio.to_thread(self._mesh, prefix)
        return list(dict.fromkeys([*local, *mesh_items]))[:100]

    @staticmethod
    def _local(prefix: str, cwd: Path) -> list[str]:
        token = prefix.rsplit(maxsplit=1)[-1] if prefix.split() else prefix
        if token.startswith(".") or "/" in token or "\\" in token:
            return [str(path.relative_to(cwd)) for path in cwd.glob(token + "*")][:50]
        path = shutil.which(token)
        candidates = [token] if path else []
        return candidates

    def _mesh(self, prefix: str) -> list[str]:
        mesh = self.mesh
        if mesh is None:
            return []
        try:
            result = mesh.query_sync(prefix, top_k=10)
        except Exception:
            return []
        return [hit.path for hit in result.hits]
