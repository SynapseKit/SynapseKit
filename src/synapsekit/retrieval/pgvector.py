from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore

if TYPE_CHECKING:
    import psycopg


class DistanceStrategy(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


class PGVectorStore(VectorStore):
    """PostgreSQL with pgvector-backed vector store. Embeds externally via SynapsekitEmbeddings."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        connection_string: str,
        table_name: str = "documents",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    ) -> None:
        self._embeddings = embedding_backend
        self._connection_string = connection_string
        self._table_name = table_name
        self._distance_strategy = distance_strategy
        self._conn: psycopg.AsyncConnection | None = None
        self._import_checked = False
        self._check_imports()

    def _check_imports(self) -> None:
        if self._import_checked:
            return
        import sys

        if "psycopg" in sys.modules and sys.modules["psycopg"] is None:
            raise ImportError(
                "psycopg and pgvector required: pip install synapsekit[pgvector]"
            ) from None
        if "pgvector.psycopg" in sys.modules and sys.modules["pgvector.psycopg"] is None:
            raise ImportError(
                "psycopg and pgvector required: pip install synapsekit[pgvector]"
            ) from None
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise ImportError(
                "psycopg and pgvector required: pip install synapsekit[pgvector]"
            ) from None
        try:
            import pgvector.psycopg  # noqa: F401
        except ImportError:
            if "pgvector.psycopg" not in sys.modules:
                raise ImportError(
                    "psycopg and pgvector required: pip install synapsekit[pgvector]"
                ) from None
        self._import_checked = True

    async def _ensure_connection(self) -> psycopg.AsyncConnection:
        self._check_imports()
        if self._conn is None:
            import psycopg

            self._conn = await psycopg.AsyncConnection.connect(self._connection_string)
            await self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await self._init_table()
        return self._conn

    async def _init_table(self) -> None:
        op_string = self._get_operator_string()
        conn = self._conn
        if conn is None:
            raise RuntimeError("PGVector connection not initialized")
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB,
                embedding vector({self._embeddings.dimension})
            )
            """
        )
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_idx
            ON {self._table_name} USING ivfflat (embedding {op_string})
            """
        )

    def _get_operator_string(self) -> str:
        if self._distance_strategy == DistanceStrategy.COSINE:
            return "cosine_ops"
        elif self._distance_strategy == DistanceStrategy.L2:
            return "l2_ops"
        else:
            return "inner_product_ops"

    def _get_select_score(self) -> str:
        if self._distance_strategy == DistanceStrategy.COSINE:
            return "1 - (embedding <=> %s) AS score"
        elif self._distance_strategy == DistanceStrategy.L2:
            return "embedding <-> %s AS score"
        else:
            return "embedding <#> %s AS score"

    def _get_search_operator(self) -> str:
        if self._distance_strategy == DistanceStrategy.COSINE:
            return "<=>"
        elif self._distance_strategy == DistanceStrategy.L2:
            return "<->"
        else:
            return "<#>"

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return
        conn = await self._ensure_connection()
        meta = metadata or [{} for _ in texts]
        vecs = await self._embeddings.embed(texts)
        import json

        for i, text in enumerate(texts):
            await conn.execute(
                f"""
                INSERT INTO {self._table_name} (text, metadata, embedding)
                VALUES (%s, %s, %s)
                """,
                (text, json.dumps(meta[i]), vecs[i].tolist()),
            )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        conn = await self._ensure_connection()
        q_vec = await self._embeddings.embed_one(query)
        score_expr = self._get_select_score()
        op = self._get_search_operator()

        where_clause = ""
        params: list = [q_vec.tolist()]
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append("metadata->>%s = %s")
                params.extend([key, str(value)])
            where_clause = "WHERE " + " AND ".join(conditions)

        query_sql = f"""
            SELECT text, metadata, {score_expr}
            FROM {self._table_name}
            {where_clause}
            ORDER BY embedding {op} %s
            LIMIT %s
        """
        params.append(q_vec.tolist())
        params.append(top_k)

        rows = await conn.fetch(query_sql, *params)
        return [
            {"text": row["text"], "score": float(row["score"]), "metadata": row["metadata"]}
            for row in rows
        ]

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support load()")
