"""CassandraVectorStore — DataStax Astra / Cassandra vector store backend."""

from __future__ import annotations

import asyncio
import json
import uuid
from functools import partial

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class CassandraVectorStore(VectorStore):
    """DataStax Astra / Apache Cassandra vector store.

    Uses ``astrapy`` when ``astra_db_id`` is provided; otherwise falls back
    to ``cassandra-driver`` with a direct contact-point connection.
    """

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        keyspace: str,
        table_name: str = "synapsekit_vec",
        session=None,
        contact_points: list[str] | None = None,
        astra_db_id: str | None = None,
        astra_token: str | None = None,
    ) -> None:
        self._embeddings = embedding_backend
        self._keyspace = keyspace
        self._table_name = table_name
        self._table_created = False
        self._dim: int | None = None
        self._mode: str  # "astra" or "cassandra"

        if astra_db_id:
            try:
                from astrapy import DataAPIClient
            except ImportError:
                raise ImportError("astrapy required: pip install synapsekit[cassandra]") from None
            self._mode = "astra"
            self._astra_db_id = astra_db_id
            self._astra_token = astra_token or ""
            from astrapy import DataAPIClient

            self._astra_client = DataAPIClient(token=self._astra_token)
            self._db = self._astra_client.get_database_by_api_endpoint(
                f"https://{astra_db_id}-{keyspace}.apps.astra.datastax.com"
            )
        else:
            try:
                from cassandra.cluster import Cluster
            except ImportError:
                raise ImportError(
                    "cassandra-driver required: pip install synapsekit[cassandra]"
                ) from None
            self._mode = "cassandra"
            if session is not None:
                self._session = session
            else:
                from cassandra.cluster import Cluster

                cluster = Cluster(contact_points=contact_points or ["127.0.0.1"])
                self._session = cluster.connect(keyspace)

    # ------------------------------------------------------------------ Astra

    def _astra_ensure_collection(self, dim: int) -> None:
        if self._table_created and self._dim == dim:
            return
        try:
            self._db.get_collection(self._table_name)
            self._table_created = True
            self._dim = dim
            return
        except Exception:
            pass
        self._db.create_collection(
            self._table_name,
            dimension=dim,
            metric="cosine",
        )
        self._table_created = True
        self._dim = dim

    def _astra_add_sync(
        self, texts: list[str], metadata: list[dict], vecs: list[list[float]]
    ) -> None:
        dim = len(vecs[0])
        self._astra_ensure_collection(dim)
        collection = self._db.get_collection(self._table_name)
        docs = [
            {
                "_id": str(uuid.uuid4()),
                "text": text,
                "metadata": meta,
                "$vector": vec,
            }
            for text, meta, vec in zip(texts, metadata, vecs, strict=True)
        ]
        collection.insert_many(docs)

    def _astra_search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        try:
            collection = self._db.get_collection(self._table_name)
            resp = collection.find(
                filter=metadata_filter or {},
                sort={"$vector": q_vec},
                limit=top_k,
                projection={"text": True, "metadata": True, "$similarity": True},
                include_similarity=True,
            )
        except Exception:
            # Collection doesn't exist yet (search before add / fresh reconnect).
            return []
        results = []
        for doc in resp:
            results.append(
                {
                    "text": doc.get("text", ""),
                    "score": float(doc.get("$similarity", 0.0)),
                    "metadata": doc.get("metadata") or {},
                }
            )
        return results

    # ------------------------------------------------------------- Cassandra

    def _cass_ensure_table(self, dim: int) -> None:
        if self._table_created and self._dim == dim:
            return
        # Cassandra 5.0 ANN search needs a typed vector column plus a
        # StorageAttachedIndex — not a plain list<float>.
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._table_name} (
                id uuid PRIMARY KEY,
                text text,
                metadata text,
                embedding vector<float, {int(dim)}>
            )
            """
        )
        self._session.execute(
            f"CREATE CUSTOM INDEX IF NOT EXISTS {self._table_name}_ann_idx "
            f"ON {self._keyspace}.{self._table_name}(embedding) "
            "USING 'StorageAttachedIndex' "
            "WITH OPTIONS = {'similarity_function': 'cosine'}"
        )
        self._table_created = True
        self._dim = dim

    def _cass_add_sync(
        self, texts: list[str], metadata: list[dict], vecs: list[list[float]]
    ) -> None:
        dim = len(vecs[0])
        self._cass_ensure_table(dim)
        stmt = self._session.prepare(
            f"INSERT INTO {self._keyspace}.{self._table_name} "
            "(id, text, metadata, embedding) VALUES (?, ?, ?, ?)"
        )
        for text, meta, vec in zip(texts, metadata, vecs, strict=True):
            self._session.execute(stmt, (uuid.uuid4(), text, json.dumps(meta), vec))

    def _cass_search_sync(
        self, q_vec: list[float], top_k: int, metadata_filter: dict | None
    ) -> list[dict]:
        # top_k lands in the LIMIT clause. Validate/cast to a positive int so it can
        # never carry injected CQL. The query vector goes in the ``ORDER BY ... ANN OF``
        # clause, where the CQL grammar requires a vector literal, not a bound marker,
        # so it cannot be parameterised; we build it from floats coerced above in
        # ``add`` (they originate from the embedding backend, never user text).
        from cassandra import InvalidRequest

        limit = int(top_k)
        if limit <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        vec_literal = "[" + ",".join(str(float(x)) for x in q_vec) + "]"
        # similarity_cosine gives the real score; the ANN clause requires a vector
        # literal (not a bound marker), so the vector is inlined from coerced floats.
        cql = (
            f"SELECT text, metadata, similarity_cosine(embedding, {vec_literal}) AS score "
            f"FROM {self._keyspace}.{self._table_name} "
            f"ORDER BY embedding ANN OF {vec_literal} LIMIT %s"
        )
        try:
            rows = self._session.execute(cql, (limit,))
        except InvalidRequest:
            # Table/index doesn't exist yet (search before add, or a fresh
            # reconnected instance with nothing indexed) — treat as no results.
            return []
        results = []
        for row in rows:
            try:
                meta = json.loads(row.metadata) if row.metadata else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            results.append({"text": row.text, "score": float(row.score), "metadata": meta})
        return results

    # --------------------------------------------------------- Public API

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
        vec_lists = [v.tolist() for v in vecs]
        loop = asyncio.get_running_loop()
        if self._mode == "astra":
            await loop.run_in_executor(None, partial(self._astra_add_sync, texts, meta, vec_lists))
        else:
            await loop.run_in_executor(None, partial(self._cass_add_sync, texts, meta, vec_lists))

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        # No early-return on an uncreated table: a reconnected instance has no
        # cached dim but rows may already exist. The mode-specific *_search_sync
        # returns [] when the table/collection genuinely doesn't exist.
        q_vec = await self._embeddings.embed_one(query)
        loop = asyncio.get_running_loop()
        if self._mode == "astra":
            return await loop.run_in_executor(
                None,
                partial(self._astra_search_sync, q_vec.tolist(), top_k, metadata_filter),
            )
        else:
            return await loop.run_in_executor(
                None,
                partial(self._cass_search_sync, q_vec.tolist(), top_k, metadata_filter),
            )
