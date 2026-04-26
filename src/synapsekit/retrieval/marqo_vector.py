"""MarqoVectorStore — Marqo neural search backend."""

from __future__ import annotations

import asyncio
import uuid
from functools import partial

from .base import VectorStore


class MarqoVectorStore(VectorStore):
    """Marqo-backed neural search vector store.

    Marqo handles embeddings natively, so embedding_backend is optional.
    """

    def __init__(
        self,
        url: str = "http://localhost:8882",
        index_name: str = "synapsekit-docs",
        embedding_backend=None,  # unused; Marqo handles embeddings
        model: str = "hf/e5-base-v2",
    ) -> None:
        try:
            import marqo
        except ImportError:
            raise ImportError("marqo required: pip install synapsekit[marqo]") from None

        self._index_name = index_name
        self._model = model
        self._index_created = False

        import marqo

        self._mq = marqo.Client(url=url)

    def _ensure_index(self) -> None:
        if self._index_created:
            return
        try:
            self._mq.index(self._index_name).get_stats()
            self._index_created = True
            return
        except Exception:
            pass
        self._mq.create_index(self._index_name, model=self._model)
        self._index_created = True

    def _add_sync(self, texts: list[str], metadata: list[dict]) -> None:
        self._ensure_index()
        docs = []
        for text, meta in zip(texts, metadata, strict=True):
            doc = {"_id": str(uuid.uuid4()), "text": text, **meta}
            docs.append(doc)
        self._mq.index(self._index_name).add_documents(docs, tensor_fields=["text"])

    def _search_sync(
        self, query: str, top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        self._ensure_index()
        resp = self._mq.index(self._index_name).search(q=query, limit=top_k)
        results = []
        for hit in resp.get("hits", []):
            text = hit.get("text", "")
            score = hit.get("_score", 0.0)
            meta = {k: v for k, v in hit.items() if k not in ("_id", "_score", "text")}
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            results.append({"text": text, "score": float(score), "metadata": meta})
        return results

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return
        meta = metadata or [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError("metadata must match texts length")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self._add_sync, texts, meta))

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._search_sync, query, top_k, metadata_filter)
        )
