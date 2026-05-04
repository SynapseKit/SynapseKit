"""ClickHouseVectorStore — ClickHouse vector store backend."""

from __future__ import annotations

import asyncio
import json
import uuid
from functools import partial

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class ClickHouseVectorStore(VectorStore):
    """ClickHouse-backed vector store using L2Distance for ANN search."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        table_name: str = "synapsekit_vec",
        user: str = "default",
        password: str = "",
    ) -> None:
        try:
            import clickhouse_connect
        except ImportError:
            raise ImportError(
                "clickhouse-connect required: pip install synapsekit[clickhouse]"
            ) from None

        self._embeddings = embedding_backend
        self._database = database
        self._table_name = table_name
        self._table_created = False
        self._dim: int | None = None

        import clickhouse_connect

        self._client = clickhouse_connect.get_client(
            host=host,
            port=port,
            database=database,
            username=user,
            password=password,
        )

    @staticmethod
    def _q(name: str) -> str:
        return "`" + name.replace("`", "``") + "`"

    def _ensure_table(self, dim: int) -> None:
        if self._table_created and self._dim == dim:
            return
        self._client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self._q(self._table_name)} (
                id String,
                text String,
                metadata String,
                embedding Array(Float32)
            ) ENGINE = MergeTree()
            ORDER BY id
            """
        )
        self._table_created = True
        self._dim = dim

    def _add_sync(self, texts: list[str], metadata: list[dict], vecs: list[list[float]]) -> None:
        dim = len(vecs[0])
        self._ensure_table(dim)
        rows = [
            [str(uuid.uuid4()), text, json.dumps(meta), vec]
            for text, meta, vec in zip(texts, metadata, vecs, strict=True)
        ]
        self._client.insert(
            self._table_name,
            rows,
            column_names=["id", "text", "metadata", "embedding"],
        )

    def _search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        vec_literal = "[" + ",".join(str(x) for x in q_vec) + "]"
        sql = (
            f"SELECT text, metadata, L2Distance(embedding, {vec_literal}) AS score "
            f"FROM {self._q(self._table_name)} "
            f"ORDER BY score ASC "
            f"LIMIT {top_k}"
        )
        result = self._client.query(sql)
        rows = result.result_rows
        out = []
        for text, raw_meta, score in rows:
            try:
                meta = json.loads(raw_meta) if raw_meta else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            out.append({"text": text, "score": float(score), "metadata": meta})
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
        if not self._table_created:
            return []
        q_vec = await self._embeddings.embed_one(query)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._search_sync, q_vec.tolist(), top_k, metadata_filter),
        )
