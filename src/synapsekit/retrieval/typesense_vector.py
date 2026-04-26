"""TypesenseVectorStore — Typesense vector search backend."""

from __future__ import annotations

import asyncio
import uuid
from functools import partial

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class TypesenseVectorStore(VectorStore):
    """Typesense-backed vector store."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        host: str = "localhost",
        port: int = 8108,
        api_key: str = "",
        collection_name: str = "synapsekit_docs",
        protocol: str = "http",
    ) -> None:
        try:
            import typesense
        except ImportError:
            raise ImportError("typesense required: pip install synapsekit[typesense]") from None

        self._embeddings = embedding_backend
        self._collection_name = collection_name
        self._collection_created = False
        self._dim: int | None = None

        import typesense

        self._ts = typesense.Client(
            {
                "nodes": [{"host": host, "port": port, "protocol": protocol}],
                "api_key": api_key,
                "connection_timeout_seconds": 10,
            }
        )

    def _ensure_collection(self, dim: int) -> None:
        if self._collection_created and self._dim == dim:
            return
        try:
            self._ts.collections[self._collection_name].retrieve()
            self._collection_created = True
            self._dim = dim
            return
        except Exception:
            pass

        self._ts.collections.create(
            {
                "name": self._collection_name,
                "fields": [
                    {"name": "text", "type": "string"},
                    {"name": "metadata", "type": "string"},
                    {
                        "name": "embedding",
                        "type": "float[]",
                        "num_dim": dim,
                    },
                ],
            }
        )
        self._collection_created = True
        self._dim = dim

    def _add_sync(self, texts: list[str], metadata: list[dict], vecs: list[list[float]]) -> None:
        import json

        dim = len(vecs[0])
        self._ensure_collection(dim)
        docs = [
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "metadata": json.dumps(meta),
                "embedding": vec,
            }
            for text, meta, vec in zip(texts, metadata, vecs, strict=True)
        ]
        self._ts.collections[self._collection_name].documents.import_(docs, {"action": "create"})

    def _search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        import json

        params = {
            "collection": self._collection_name,
            "q": "*",
            "vector_query": f"embedding:([{','.join(str(x) for x in q_vec)}], k:{top_k})",
            "per_page": top_k,
        }
        resp = self._ts.multi_search.perform({"searches": [params]}, {})
        hits = resp["results"][0].get("hits", [])
        results = []
        for hit in hits:
            doc = hit["document"]
            text = doc.get("text", "")
            score = hit.get("vector_distance", 0.0)
            raw_meta = doc.get("metadata", "{}")
            try:
                meta = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
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
        if self._dim is None:
            return []
        q_vec = await self._embeddings.embed_one(query)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._search_sync, q_vec.tolist(), top_k, metadata_filter),
        )
