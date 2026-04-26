"""VespaVectorStore — Vespa-backed vector store backend."""

from __future__ import annotations

import asyncio
import json
import uuid
from functools import partial
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class VespaVectorStore(VectorStore):
    """Vespa-backed vector store using the Vespa HTTP Document/Search API."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        url: str = "http://localhost:8080",
        application: str = "default",
        schema: str = "synapsekit",
        namespace: str = "default",
        content_cluster: str = "content",
    ) -> None:
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError("requests required: pip install synapsekit[vespa]") from None

        self._embeddings = embedding_backend
        self._url = url.rstrip("/")
        self._application = application
        self._schema = schema
        self._namespace = namespace
        self._content_cluster = content_cluster

    def _get_requests(self):  # type: ignore[return]
        import requests

        return requests

    def _doc_url(self, doc_id: str) -> str:
        return (
            f"{self._url}/document/v1/{self._application}"
            f"/{self._schema}/docid/{doc_id}"
        )

    def _add_sync(self, texts: list[str], metadata: list[dict], vecs: list[list[float]]) -> None:
        requests = self._get_requests()
        for text, meta, vec in zip(texts, metadata, vecs, strict=True):
            doc_id = str(uuid.uuid4())
            body: dict[str, Any] = {
                "fields": {
                    "text": text,
                    "embedding": {"values": vec},
                    "metadata": json.dumps(meta),
                }
            }
            resp = requests.post(self._doc_url(doc_id), json=body, timeout=30)
            resp.raise_for_status()

    def _search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        requests = self._get_requests()
        yql = (
            f"select * from sources {self._schema} "
            f"where ({{targetHits: {top_k}}}nearestNeighbor(embedding, query_embedding))"
        )
        body: dict[str, Any] = {
            "yql": yql,
            "hits": top_k,
            "ranking.profile": "semantic",
            "input.query(query_embedding)": q_vec,
        }
        resp = requests.post(f"{self._url}/search/", json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for hit in data.get("root", {}).get("children", []):
            fields = hit.get("fields", {})
            text = fields.get("text", "")
            score = hit.get("relevance", 0.0)
            raw_meta = fields.get("metadata", "{}")
            try:
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except (json.JSONDecodeError, TypeError):
                meta = {}
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            results.append({"text": text, "score": float(score), "metadata": meta})
        return results[:top_k]

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
