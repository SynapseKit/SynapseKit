"""PGVectorStore — PostgreSQL pgvector-backed vector store backend."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

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

    The embedding dimension is taken from the first vector produced at ``add()``
    time, so any embeddings backend works (no ``.dimension`` attribute needed).
    The table and index are created on the first ``add()``; ``search()`` before
    any documents exist returns ``[]``.

    Prerequisites:
        - PostgreSQL with the pgvector extension available
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
        self._table_created = False
        self._dim: int | None = None

    async def _ensure_connection(self) -> psycopg.AsyncConnection:
        if self._conn is None:
            import psycopg
            from pgvector.psycopg import register_vector_async

            conn = await psycopg.AsyncConnection.connect(
                self._connection_string, autocommit=True
            )
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Register the vector type adapters so numpy arrays / lists round-trip
            # to the pgvector ``vector`` column. Must run after CREATE EXTENSION.
            await register_vector_async(conn)
            self._conn = conn
        return self._conn

    async def _ensure_table(self, dim: int) -> None:
        if self._table_created and self._dim == dim:
            return
        from psycopg import sql

        conn = await self._ensure_connection()
        op_class = self._get_operator_class()

        await conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector({dim})
                )
                """
            ).format(table=sql.Identifier(self._table_name), dim=sql.Literal(dim))
        )
        await conn.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {idx} ON {table} "
                "USING hnsw (embedding {op})"
            ).format(
                idx=sql.Identifier(f"{self._table_name}_embedding_idx"),
                table=sql.Identifier(self._table_name),
                op=sql.SQL(op_class),
            )
        )
        self._table_created = True
        self._dim = dim

    def _get_operator_class(self) -> str:
        if self._distance_strategy == DistanceStrategy.COSINE:
            return "vector_cosine_ops"
        elif self._distance_strategy == DistanceStrategy.L2:
            return "vector_l2_ops"
        return "vector_ip_ops"

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
        from psycopg.types.json import Jsonb

        meta = metadata or [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError("metadata must match texts length")

        vecs = await self._embeddings.embed(texts)
        dim = int(vecs.shape[1]) if hasattr(vecs, "shape") else len(vecs[0])
        conn = await self._ensure_connection()
        await self._ensure_table(dim)

        insert = sql.SQL(
            "INSERT INTO {} (text, metadata, embedding) VALUES (%s, %s, %s)"
        ).format(sql.Identifier(self._table_name))
        for i, text in enumerate(texts):
            await conn.execute(insert, (text, Jsonb(meta[i]), vecs[i]))

    async def _detect_existing_table(self, conn: psycopg.AsyncConnection) -> bool:
        cur = await conn.execute("SELECT to_regclass(%s)", (self._table_name,))
        row = await cur.fetchone()
        return bool(row and row[0] is not None)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        from psycopg import sql

        conn = await self._ensure_connection()
        if not self._table_created and not await self._detect_existing_table(conn):
            return []
        self._table_created = True

        q_vec = await self._embeddings.embed_one(query)
        op = self._get_distance_operator()

        where_parts: list[sql.Composable] = []
        params: list[Any] = []
        if metadata_filter:
            for key, value in metadata_filter.items():
                where_parts.append(sql.SQL("metadata->>%s = %s"))
                params.extend([key, str(value)])

        if self._distance_strategy == DistanceStrategy.COSINE:
            score_expr = sql.SQL("1 - (embedding {} %s) AS score").format(sql.SQL(op))
        else:
            score_expr = sql.SQL("embedding {} %s AS score").format(sql.SQL(op))

        query_parts: list[sql.Composable] = [
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
        all_params = [q_vec, *params, q_vec, top_k]

        async with conn.cursor() as cur:
            await cur.execute(query_sql, all_params)
            rows = await cur.fetchall()
            col_names = [desc[0] for desc in cur.description or []]

        text_i = col_names.index("text")
        meta_i = col_names.index("metadata")
        score_i = col_names.index("score")
        return [
            {
                "text": row[text_i],
                "score": float(row[score_i]),
                "metadata": _as_dict(row[meta_i]),
            }
            for row in rows
        ]


def _as_dict(value: Any) -> dict:
    """psycopg returns jsonb as a dict already; tolerate a str or None too."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    import json

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
