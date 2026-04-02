"""PGVectorStore — PostgreSQL pgvector-backed vector store backend."""

from __future__ import annotations

import json
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
    """PostgreSQL with pgvector-backed vector store. Embeds externally via SynapsekitEmbeddings.

    Prerequisites:
        - PostgreSQL with the pgvector extension installed
        - The database user must have permission to run ``CREATE EXTENSION``
          (requires ``SUPERUSER`` or ``rds_superuser`` on managed PostgreSQL)

    Example::

        store = PGVectorStore(
            embedding_backend=embeddings,
            connection_string="postgresql://user:pass@localhost/db",
        )
        await store.add(["hello world"], metadata=[{"source": "demo"}])
        results = await store.search("hello")
    """

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        connection_string: str,
        table_name: str = "documents",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    ) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise ImportError(
                "psycopg and pgvector required: pip install synapsekit[pgvector]"
            ) from None
        try:
            import pgvector.psycopg  # noqa: F401
        except ImportError:
            raise ImportError(
                "psycopg and pgvector required: pip install synapsekit[pgvector]"
            ) from None

        self._embeddings = embedding_backend
        self._connection_string = connection_string
        self._table_name = table_name
        self._distance_strategy = distance_strategy
        self._conn: psycopg.AsyncConnection | None = None

    async def _ensure_connection(self) -> psycopg.AsyncConnection:
        if self._conn is None:
            import psycopg

            self._conn = await psycopg.AsyncConnection.connect(self._connection_string)
            await self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await self._init_table()
        return self._conn

    async def _init_table(self) -> None:
        from psycopg import sql

        conn = self._conn
        if conn is None:
            raise RuntimeError("PGVector connection not initialized")

        op_string = self._get_operator_string()
        dim = self._embeddings.dimension

        await conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({})
                )
                """
            ).format(sql.Identifier(self._table_name), sql.Literal(dim))
        )
        await conn.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {idx}
                ON {table} USING ivfflat (embedding {op})
                """
            ).format(
                idx=sql.Identifier(f"{self._table_name}_embedding_idx"),
                table=sql.Identifier(self._table_name),
                op=sql.SQL(op_string),
            )
        )

    def _get_operator_string(self) -> str:
        if self._distance_strategy == DistanceStrategy.COSINE:
            return "cosine_ops"
        elif self._distance_strategy == DistanceStrategy.L2:
            return "l2_ops"
        return "inner_product_ops"

    def _get_distance_operator(self) -> str:
        if self._distance_strategy == DistanceStrategy.COSINE:
            return "<=>"
        elif self._distance_strategy == DistanceStrategy.L2:
            return "<->"
        return "<#>"

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return
        from psycopg import sql

        conn = await self._ensure_connection()
        meta = metadata or [{} for _ in texts]
        vecs = await self._embeddings.embed(texts)

        for i, text in enumerate(texts):
            await conn.execute(
                sql.SQL("INSERT INTO {} (text, metadata, embedding) VALUES (%s, %s, %s)").format(
                    sql.Identifier(self._table_name)
                ),
                (text, json.dumps(meta[i]), vecs[i].tolist()),
            )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        from psycopg import sql

        conn = await self._ensure_connection()
        q_vec = await self._embeddings.embed_one(query)
        op = self._get_distance_operator()

        where_parts: list[sql.Composable] = []
        params: list = []
        if metadata_filter:
            for key, value in metadata_filter.items():
                where_parts.append(sql.SQL("metadata->>%s = %s"))
                params.extend([key, str(value)])

        if self._distance_strategy == DistanceStrategy.COSINE:
            score_expr = sql.SQL("1 - (embedding {} %s) AS score").format(sql.SQL("<=>"))
        elif self._distance_strategy == DistanceStrategy.L2:
            score_expr = sql.SQL("embedding {} %s AS score").format(sql.SQL("<->"))
        else:
            score_expr = sql.SQL("embedding {} %s AS score").format(sql.SQL("<#>"))

        score_params = [q_vec.tolist()]
        order_params = [q_vec.tolist(), top_k]

        query_parts = [
            sql.SQL("SELECT text, metadata, "),
            score_expr,
            sql.SQL(" FROM "),
            sql.Identifier(self._table_name),
        ]
        if where_parts:
            query_parts.append(sql.SQL(" WHERE "))
            query_parts.append(sql.SQL(" AND ").join(where_parts))

        query_parts.append(sql.SQL(" ORDER BY embedding {} %s LIMIT %s").format(sql.SQL(op)))

        query_sql = sql.Composed(query_parts)
        all_params = score_params + params + order_params

        async with await conn.cursor() as cur:
            await cur.execute(query_sql, all_params)
            rows = await cur.fetchall()
            col_names = [desc[0] for desc in cur.description or []]

        return [
            {
                "text": row[col_names.index("text")],
                "score": float(row[col_names.index("score")]),
                "metadata": json.loads(row[col_names.index("metadata")] or "{}"),
            }
            for row in rows
        ]
