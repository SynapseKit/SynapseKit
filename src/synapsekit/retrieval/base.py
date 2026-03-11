from __future__ import annotations

from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Abstract base class for all vector store backends."""

    @abstractmethod
    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None: ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[dict]: ...

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support load()")
