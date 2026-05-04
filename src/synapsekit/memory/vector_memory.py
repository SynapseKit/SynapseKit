"""VectorConversationMemory: vector-backed semantic conversation memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..embeddings.backend import SynapsekitEmbeddings


class VectorConversationMemory:
    """Vector-backed conversation memory with semantic retrieval.

    Stores each message as an embedding. On ``get_context(query, k=5)``,
    retrieves the *k* most semantically similar past turns.

    Usage::

        mem = VectorConversationMemory(embedding_backend=embeddings)
        await mem.add_message("user", "Hello, my name is Alice.")
        context = await mem.get_context("Who am I?")
    """

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        max_tokens: int = 2000,
        k: int = 5,
    ) -> None:
        from ..retrieval.vectorstore import InMemoryVectorStore

        self._embeddings = embedding_backend
        self._max_tokens = max_tokens
        self._k = k
        self._store = InMemoryVectorStore(embedding_backend)
        self._messages: list[dict] = []

    async def add_message(self, role: str, content: str) -> None:
        """Embed and store a message."""
        self._messages.append({"role": role, "content": content})
        text = f"{role}: {content}"
        await self._store.add([text], [{"role": role, "index": len(self._messages) - 1}])

    async def get_context(self, query: str, k: int | None = None) -> list[dict]:
        """Return the k most semantically similar past messages for query."""
        top_k = k if k is not None else self._k
        results = await self._store.search(query, top_k=top_k)
        context: list[dict] = []
        for r in results:
            idx = r["metadata"].get("index")
            if idx is not None and idx < len(self._messages):
                context.append(self._messages[idx])
        return context

    def clear(self) -> None:
        """Clear all stored messages and vectors."""
        from ..retrieval.vectorstore import InMemoryVectorStore

        self._messages.clear()
        self._store = InMemoryVectorStore(self._embeddings)

    @property
    def messages(self) -> list[dict]:
        """Return a copy of all stored messages."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
