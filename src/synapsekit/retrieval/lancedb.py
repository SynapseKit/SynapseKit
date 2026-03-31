"""LanceDBVectorStore — LanceDB-backed vector store backend."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore

logger = logging.getLogger(__name__)


class LanceDBVectorStore(VectorStore):
    """LanceDB-backed vector store. Embeds externally via SynapsekitEmbeddings."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        uri: str = ".lancedb",
        table_name: str = "synapsekit",
        text_field: str = "text",
        vector_field: str = "embedding",
    ) -> None:
        try:
            import lancedb
        except ImportError:
            raise ImportError("lancedb required: pip install synapsekit[lancedb]") from None

        self._embeddings = embedding_backend
        self._table_name = table_name
        self._text_field = text_field
        self._vector_field = vector_field
        self._db = lancedb.connect(uri)
        self._table: Any | None = None
        self._fts_ready = False

    def _open_table(self) -> Any | None:
        if self._table is not None:
            return self._table
        try:
            self._table = self._db.open_table(self._table_name)
        except (FileNotFoundError, KeyError):
            return None
        except Exception as exc:
            logger.debug("LanceDBVectorStore: could not open table %r — %s", self._table_name, exc)
            return None
        return self._table

    def _ensure_fts_index(self, table: Any) -> None:
        if self._fts_ready or not hasattr(table, "create_fts_index"):
            return
        try:
            table.create_fts_index(self._text_field)
        except TypeError:
            try:
                table.create_fts_index([self._text_field])
            except Exception:
                return
        except Exception:
            return
        self._fts_ready = True

    def _build_filter_expr(self, metadata_filter: dict[str, Any] | None) -> str | None:
        if not metadata_filter:
            return None
        clauses = []
        for key, value in metadata_filter.items():
            clauses.append(f"{key} == {json.dumps(value)}")
        return " and ".join(clauses)

    @staticmethod
    def _normalize_row(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        if hasattr(row, "to_dict"):
            raw = row.to_dict()
            return raw if isinstance(raw, dict) else dict(raw)
        if hasattr(row, "items"):
            return dict(row.items())
        return {
            key: getattr(row, key)
            for key in dir(row)
            if not key.startswith("_") and not callable(getattr(row, key))
        }

    @staticmethod
    def _extract_metadata(row: dict[str, Any]) -> dict[str, Any]:
        reserved = {"text", "score", "distance", "_score", "_distance", "embedding", "vector"}
        return {key: value for key, value in row.items() if key not in reserved}

    def _create_table(self, rows: list[dict[str, Any]]) -> Any:
        table = self._db.create_table(self._table_name, data=rows)
        self._table = table
        self._ensure_fts_index(table)
        return table

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
        rows = []
        for text, vector, extra in zip(texts, vecs, meta, strict=True):
            rows.append({self._text_field: text, self._vector_field: vector.tolist(), **extra})

        table = self._open_table()
        if table is None:
            self._create_table(rows)
            return

        self._ensure_fts_index(table)
        table.add(rows)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        table = self._open_table()
        if table is None:
            return []

        # Embed the query and search by vector, not raw text string
        q_vec = await self._embeddings.embed_one(query)
        builder = table.search(q_vec.tolist())

        expr = self._build_filter_expr(metadata_filter)
        if expr and hasattr(builder, "where"):
            builder = builder.where(expr)
        if hasattr(builder, "limit"):
            builder = builder.limit(top_k)

        results = builder.to_list() if hasattr(builder, "to_list") else builder
        out: list[dict] = []
        for row in results or []:
            item = self._normalize_row(row)
            score = item.get(
                "score", item.get("_score", item.get("distance", item.get("_distance", 0.0)))
            )
            out.append(
                {
                    "text": item.get(self._text_field, item.get("text", "")),
                    "score": float(score),
                    "metadata": self._extract_metadata(item),
                }
            )
        return out
