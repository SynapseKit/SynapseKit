"""DuckDBVectorStore — DuckDB VSS extension vector store backend."""

from __future__ import annotations

import asyncio
import json
import uuid
from functools import partial

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class DuckDBVectorStore(VectorStore):
    """DuckDB VSS extension-backed vector store (array_cosine_similarity)."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        db_path: str = ":memory:",
        table_name: str = "synapsekit_vec",
    ) -> None:
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "duckdb required: pip install synapsekit[duckdb-vector]"
            ) from None

        self._embeddings = embedding_backend
        self._table_name = table_name
        self._dim: int | None = None
        self._table_created = False

        import duckdb

        self._conn = duckdb.connect(db_path)
        self._install_vss()

    def _install_vss(self) -> None:
        try:
            self._conn.execute("INSTALL vss; LOAD vss;")
        except Exception:
            # VSS may already be loaded or not available; array_cosine_similarity
            # is available as a built-in in recent DuckDB versions.
            pass

    @staticmethod
    def _q(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _ensure_table(self, dim: int) -> None:
        if self._table_created and self._dim == dim:
            return
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._q(self._table_name)} (
                id VARCHAR PRIMARY KEY,
                text VARCHAR NOT NULL,
                metadata VARCHAR,
                embedding FLOAT[{dim}]
            )
            """
        )
        self._table_created = True
        self._dim = dim

    def _add_sync(self, texts: list[str], metadata: list[dict], vecs: list[list[float]]) -> None:
        dim = len(vecs[0])
        self._ensure_table(dim)
        for text, meta, vec in zip(texts, metadata, vecs, strict=True):
            doc_id = str(uuid.uuid4())
            self._conn.execute(
                f"INSERT INTO {self._q(self._table_name)} (id, text, metadata, embedding) "
                "VALUES (?, ?, ?, ?::FLOAT[])",
                [doc_id, text, json.dumps(meta), vec],
            )

    def _search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        rows = self._conn.execute(
            f"""
            SELECT text, metadata,
                   array_cosine_similarity(embedding, ?::FLOAT[]) AS score
            FROM {self._q(self._table_name)}
            ORDER BY score DESC
            LIMIT ?
            """,
            [q_vec, top_k],
        ).fetchall()

        results = []
        for text, raw_meta, score in rows:
            try:
                meta = json.loads(raw_meta) if raw_meta else {}
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
        if not self._table_created:
            return []
        q_vec = await self._embeddings.embed_one(query)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(self._search_sync, q_vec.tolist(), top_k, metadata_filter),
        )
