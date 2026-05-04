"""ElasticsearchVectorStore — Elasticsearch dense_vector store backend."""

from __future__ import annotations

import asyncio
import uuid
from functools import partial

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class ElasticsearchVectorStore(VectorStore):
    """Elasticsearch dense_vector knn search vector store."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        url: str = "http://localhost:9200",
        index_name: str = "synapsekit_vec",
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        dims: int | None = None,
    ) -> None:
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(
                "elasticsearch required: pip install synapsekit[elasticsearch-vector]"
            ) from None

        self._embeddings = embedding_backend
        self._url = url
        self._index_name = index_name
        self._dims = dims
        self._index_created = False

        from elasticsearch import Elasticsearch

        kwargs: dict = {"hosts": [url]}
        if api_key:
            kwargs["api_key"] = api_key
        elif username and password:
            kwargs["basic_auth"] = (username, password)
        self._es = Elasticsearch(**kwargs)

    def _ensure_index(self, dim: int) -> None:
        if self._index_created and self._dims == dim:
            return
        if self._es.indices.exists(index=self._index_name):
            self._index_created = True
            self._dims = dim
            return
        self._es.indices.create(
            index=self._index_name,
            body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "metadata": {"type": "object"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": dim,
                            "index": True,
                            "similarity": "cosine",
                        },
                    }
                }
            },
        )
        self._index_created = True
        self._dims = dim

    def _add_sync(self, texts: list[str], metadata: list[dict], vecs: list[list[float]]) -> None:
        dim = len(vecs[0])
        self._ensure_index(dim)
        for text, meta, vec in zip(texts, metadata, vecs, strict=True):
            doc_id = str(uuid.uuid4())
            self._es.index(
                index=self._index_name,
                id=doc_id,
                document={"text": text, "metadata": meta, "embedding": vec},
            )
        self._es.indices.refresh(index=self._index_name)

    def _search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        query: dict = {
            "knn": {
                "field": "embedding",
                "query_vector": q_vec,
                "k": top_k,
                "num_candidates": top_k * 10,
            }
        }
        resp = self._es.search(index=self._index_name, body=query, size=top_k)
        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            meta = src.get("metadata") or {}
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            results.append(
                {
                    "text": src.get("text", ""),
                    "score": float(hit["_score"]),
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
        if not self._index_created and self._dims is None:
            return []
        q_vec = await self._embeddings.embed_one(query)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._search_sync, q_vec.tolist(), top_k, metadata_filter),
        )
