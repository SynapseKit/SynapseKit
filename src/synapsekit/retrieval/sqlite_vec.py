from __future__ import annotations

import json
import os
import sqlite3
from contextlib import suppress
from typing import Any

import numpy as np

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class SQLiteVecStore(VectorStore):
    """sqlite-vec backed vector store (fully embedded, zero server)."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        db_path: str = ":memory:",
        table_name: str = "synapsekit_vec",
    ) -> None:
        try:
            import sqlite_vec
        except ImportError:
            raise ImportError("sqlite-vec required: pip install synapsekit[sqlite-vec]") from None

        self._sqlite_vec = sqlite_vec
        self._embeddings = embedding_backend
        self._db_path = db_path
        self._table_name = table_name
        self._meta_table = f"{table_name}__meta"
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._dim: int | None = None

        self._load_extension()
        self._ensure_meta_table()
        self._dim = self._read_dim()

    @staticmethod
    def _q(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _load_extension(self) -> None:
        if hasattr(self._conn, "enable_load_extension"):
            self._conn.enable_load_extension(True)
        try:
            self._sqlite_vec.load(self._conn)
        finally:
            if hasattr(self._conn, "enable_load_extension"):
                self._conn.enable_load_extension(False)

    def _ensure_meta_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._q(self._meta_table)} (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                dim INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def _read_dim(self) -> int | None:
        row = self._conn.execute(
            f"SELECT dim FROM {self._q(self._meta_table)} WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return int(row[0])

    def _write_dim(self, dim: int) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {self._q(self._meta_table)} (id, dim)
            VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET dim = excluded.dim
            """,
            (dim,),
        )
        self._conn.commit()

    def _table_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?",
            (self._table_name,),
        ).fetchone()
        return row is not None

    def _ensure_vector_table(self, dim: int) -> None:
        if self._dim is not None and self._dim != dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self._dim}, got {dim}")

        if self._table_exists():
            self._dim = dim
            self._write_dim(dim)
            return

        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE {self._q(self._table_name)} USING vec0(
                embedding float[{dim}],
                +text text,
                +metadata text
            )
            """
        )
        self._conn.commit()
        self._dim = dim
        self._write_dim(dim)

    @staticmethod
    def _serialize_vec(vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

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
        if vecs.ndim != 2:
            raise ValueError("embed() must return shape (N, D)")

        self._ensure_vector_table(int(vecs.shape[1]))

        with self._conn:
            for text, vector, row_meta in zip(texts, vecs, meta, strict=True):
                self._conn.execute(
                    f"INSERT INTO {self._q(self._table_name)} (embedding, text, metadata) VALUES (?, ?, ?)",
                    (self._serialize_vec(vector), text, json.dumps(row_meta)),
                )

    @staticmethod
    def _meta_matches(meta: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
        if not metadata_filter:
            return True
        return all(meta.get(k) == v for k, v in metadata_filter.items())

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if top_k <= 0 or not self._table_exists():
            return []

        q_vec = await self._embeddings.embed_one(query)
        candidate_k = top_k if not metadata_filter else max(top_k * 5, top_k)

        rows = self._conn.execute(
            f"""
            SELECT rowid, text, metadata, distance
            FROM {self._q(self._table_name)}
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (self._serialize_vec(q_vec), candidate_k),
        ).fetchall()

        out: list[dict] = []
        for row in rows:
            if isinstance(row, tuple):
                _rowid, text_val, metadata_raw, distance_val = row
            else:
                text_val = row["text"]
                metadata_raw = row["metadata"]
                distance_val = row["distance"]

            metadata_obj = json.loads(metadata_raw) if metadata_raw else {}
            if not self._meta_matches(metadata_obj, metadata_filter):
                continue
            out.append(
                {
                    "text": text_val,
                    "score": float(distance_val),
                    "metadata": metadata_obj,
                }
            )
            if len(out) >= top_k:
                break
        return out

    def save(self, path: str) -> None:
        if not self._table_exists():
            raise ValueError("Nothing to save — store is empty.")

        dest = sqlite3.connect(path)
        try:
            self._conn.backup(dest)
        finally:
            dest.close()

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with suppress(Exception):
            self._conn.close()

        self._db_path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._load_extension()
        self._ensure_meta_table()
        self._dim = self._read_dim()

    def __len__(self) -> int:
        if not self._table_exists():
            return 0
        row = self._conn.execute(f"SELECT COUNT(*) FROM {self._q(self._table_name)}").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
