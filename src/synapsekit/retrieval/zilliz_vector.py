"""ZillizVectorStore — Zilliz Cloud (managed Milvus) vector store backend."""

from __future__ import annotations

import asyncio
from functools import partial

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class ZillizVectorStore(VectorStore):
    """Zilliz Cloud (managed Milvus) vector store using MilvusClient."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        uri: str,
        token: str,
        collection_name: str = "synapsekit_docs",
        dim: int | None = None,
    ) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError("pymilvus required: pip install synapsekit[zilliz]") from None

        self._embeddings = embedding_backend
        self._collection_name = collection_name
        self._dim = dim
        self._collection_created = False

        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=uri, token=token)

    def _ensure_collection(self, dim: int) -> None:
        if self._collection_created and self._dim == dim:
            return
        if self._client.has_collection(self._collection_name):
            self._collection_created = True
            self._dim = dim
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            dimension=dim,
            auto_id=True,
            metric_type="COSINE",
        )
        self._collection_created = True
        self._dim = dim

    def _add_sync(self, texts: list[str], metadata: list[dict], vecs: list[list[float]]) -> None:
        dim = len(vecs[0])
        self._ensure_collection(dim)
        data = [
            {"vector": vec, "text": text, "metadata": str(meta)}
            for text, meta, vec in zip(texts, metadata, vecs, strict=True)
        ]
        self._client.insert(collection_name=self._collection_name, data=data)

    def _search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        import json

        results = self._client.search(
            collection_name=self._collection_name,
            data=[q_vec],
            limit=top_k,
            output_fields=["text", "metadata"],
        )
        out = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            text = entity.get("text", "")
            score = float(hit.get("distance", 0.0))
            raw_meta = entity.get("metadata", "{}")
            try:
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except (json.JSONDecodeError, TypeError):
                meta = {}
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            out.append({"text": text, "score": score, "metadata": meta})
        return out

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
        if not self._collection_created and self._dim is None:
            return []
        q_vec = await self._embeddings.embed_one(query)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._search_sync, q_vec.tolist(), top_k, metadata_filter),
        )
