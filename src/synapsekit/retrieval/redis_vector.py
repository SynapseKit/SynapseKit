"""RedisVectorStore — RediSearch-backed vector store backend."""

from __future__ import annotations

import asyncio
import json
import struct
import uuid
from functools import partial

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


def _vec_to_bytes(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class RedisVectorStore(VectorStore):
    """RediSearch HNSW-backed vector store."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        url: str = "redis://localhost:6379",
        index_name: str = "synapsekit_idx",
        distance_metric: str = "COSINE",
    ) -> None:
        try:
            import redis  # noqa: F401
        except ImportError:
            raise ImportError("redis required: pip install synapsekit[redis-vector]") from None

        self._embeddings = embedding_backend
        self._url = url
        self._index_name = index_name
        self._distance_metric = distance_metric
        self._index_created = False
        self._dim: int | None = None

        import redis as _redis

        self._client = _redis.from_url(url, decode_responses=False)

    def _ensure_index(self, dim: int) -> None:
        if self._index_created and self._dim == dim:
            return
        try:
            self._client.execute_command("FT.INFO", self._index_name)
            self._index_created = True
            self._dim = dim
            return
        except Exception:
            pass

        self._client.execute_command(
            "FT.CREATE",
            self._index_name,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            f"{self._index_name}:",
            "SCHEMA",
            "text",
            "TEXT",
            "metadata",
            "TEXT",
            "embedding",
            "VECTOR",
            "HNSW",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(dim),
            "DISTANCE_METRIC",
            self._distance_metric,
        )
        self._index_created = True
        self._dim = dim

    def _add_sync(self, texts: list[str], metadata: list[dict], vecs: list[list[float]]) -> None:
        dim = len(vecs[0])
        self._ensure_index(dim)
        pipe = self._client.pipeline(transaction=False)
        for text, meta, vec in zip(texts, metadata, vecs, strict=True):
            doc_id = str(uuid.uuid4())
            key = f"{self._index_name}:{doc_id}"
            pipe.hset(
                key,
                mapping={
                    "text": text.encode(),
                    "metadata": json.dumps(meta).encode(),
                    "embedding": _vec_to_bytes(vec),
                },
            )
        pipe.execute()

    def _search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        query_bytes = _vec_to_bytes(q_vec)
        query = f"*=>[KNN {top_k} @embedding $vec AS score]"
        raw = self._client.execute_command(
            "FT.SEARCH",
            self._index_name,
            query,
            "PARAMS",
            "2",
            "vec",
            query_bytes,
            "SORTBY",
            "score",
            "DIALECT",
            "2",
            "LIMIT",
            "0",
            str(top_k),
        )
        results = []
        # raw: [total, key, [field, value, ...], key, ...]
        i = 1
        while i < len(raw):
            i += 1  # skip key
            fields_list = raw[i]
            i += 1
            fields: dict = {}
            for j in range(0, len(fields_list), 2):
                k_raw = fields_list[j]
                v_raw = fields_list[j + 1]
                k = k_raw.decode() if isinstance(k_raw, bytes) else k_raw
                v = v_raw.decode() if isinstance(v_raw, bytes) else v_raw
                fields[k] = v
            text = fields.get("text", "")
            score_raw = fields.get("score", "0")
            try:
                score = float(score_raw)
            except (ValueError, TypeError):
                score = 0.0
            try:
                meta = json.loads(fields.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                meta = {}
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            results.append({"text": text, "score": score, "metadata": meta})
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
