"""SupabaseVectorStore — Supabase pgvector store backend."""

from __future__ import annotations

import asyncio
from functools import partial

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class SupabaseVectorStore(VectorStore):
    """Supabase pgvector-backed vector store."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        url: str,
        key: str,
        table_name: str = "synapsekit_documents",
        query_fn: str = "match_documents",
    ) -> None:
        try:
            from supabase import create_client
        except ImportError:
            raise ImportError(
                "supabase required: pip install synapsekit[supabase-vector]"
            ) from None

        self._embeddings = embedding_backend
        self._table_name = table_name
        self._query_fn = query_fn

        from supabase import create_client

        self._client = create_client(url, key)

    def _add_sync(self, texts: list[str], metadata: list[dict], vecs: list[list[float]]) -> None:
        rows = [
            {"content": text, "metadata": meta, "embedding": vec}
            for text, meta, vec in zip(texts, metadata, vecs, strict=True)
        ]
        self._client.table(self._table_name).insert(rows).execute()

    def _search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        params: dict = {"query_embedding": q_vec, "match_count": top_k}
        resp = self._client.rpc(self._query_fn, params).execute()
        results = []
        for row in resp.data or []:
            meta = row.get("metadata") or {}
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            results.append(
                {
                    "text": row.get("content", ""),
                    "score": float(row.get("similarity", 0.0)),
                    "metadata": meta,
                }
            )
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
        vecs = await self._embeddings.embed(texts)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(self._add_sync, texts, meta, [v.tolist() for v in vecs]),
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        q_vec = await self._embeddings.embed_one(query)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._search_sync, q_vec.tolist(), top_k, metadata_filter),
        )
