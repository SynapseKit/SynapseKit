"""MilvusVectorStore — Milvus-backed vector store backend."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class MilvusIndexType(str, Enum):
    IVF_FLAT = "IVF_FLAT"
    HNSW = "HNSW"


class MilvusVectorStore(VectorStore):
    """Milvus-backed vector store. Embeds externally via SynapsekitEmbeddings."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        collection_name: str = "synapsekit",
        uri: str = "http://localhost:19530",
        token: str | None = None,
        user: str | None = None,
        password: str | None = None,
        db_name: str | None = None,
        index_type: MilvusIndexType | str = MilvusIndexType.IVF_FLAT,
        metric_type: str = "COSINE",
        nlist: int = 128,
        nprobe: int = 8,
        m: int = 16,
        ef_construction: int = 200,
        ef: int = 64,
    ) -> None:
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError:
            raise ImportError("pymilvus required: pip install synapsekit[milvus]") from None

        self._embeddings = embedding_backend
        self._collection_name = collection_name
        self._index_type = (
            index_type if isinstance(index_type, MilvusIndexType) else MilvusIndexType(index_type)
        )
        self._metric_type = metric_type.upper()
        self._nlist = nlist
        self._nprobe = nprobe
        self._m = m
        self._ef_construction = ef_construction
        self._ef = ef
        self._data_type = DataType

        # Only pass optional connection params when explicitly provided
        kwargs: dict[str, Any] = {}
        if token is not None:
            kwargs["token"] = token
        if user is not None:
            kwargs["user"] = user
        if password is not None:
            kwargs["password"] = password
        if db_name is not None:
            kwargs["db_name"] = db_name

        self._client = MilvusClient(uri=uri, **kwargs)

    def _build_index_params(self) -> dict[str, Any]:
        if self._index_type == MilvusIndexType.IVF_FLAT:
            return {
                "index_type": "IVF_FLAT",
                "metric_type": self._metric_type,
                "params": {"nlist": self._nlist},
            }
        return {
            "index_type": "HNSW",
            "metric_type": self._metric_type,
            "params": {"M": self._m, "efConstruction": self._ef_construction},
        }

    def _build_search_params(self) -> dict[str, Any]:
        if self._index_type == MilvusIndexType.IVF_FLAT:
            return {"metric_type": self._metric_type, "params": {"nprobe": self._nprobe}}
        return {"metric_type": self._metric_type, "params": {"ef": self._ef}}

    def _ensure_collection(self, dimension: int) -> None:
        if self._client.has_collection(collection_name=self._collection_name):
            self._client.load_collection(collection_name=self._collection_name)
            return

        # auto_id=True — Milvus assigns IDs automatically; do NOT add a manual primary field
        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(
            field_name="text",
            datatype=self._data_type.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name="embedding",
            datatype=self._data_type.FLOAT_VECTOR,
            dim=dimension,
        )

        self._client.create_collection(collection_name=self._collection_name, schema=schema)
        self._client.create_index(
            collection_name=self._collection_name,
            field_name="embedding",
            index_params=self._build_index_params(),
        )
        self._client.load_collection(collection_name=self._collection_name)

    @staticmethod
    def _build_expr(metadata_filter: dict[str, Any] | None) -> str | None:
        if not metadata_filter:
            return None
        clauses = []
        for key, value in metadata_filter.items():
            clauses.append(f"{key} == {json.dumps(value)}")
        return " and ".join(clauses)

    @staticmethod
    def _entity_to_dict(entity: Any) -> dict[str, Any]:
        if entity is None:
            return {}
        if isinstance(entity, dict):
            return entity
        if hasattr(entity, "to_dict"):
            raw = entity.to_dict()
            return raw if isinstance(raw, dict) else dict(raw)
        if hasattr(entity, "items"):
            return dict(entity.items())
        return {
            k: getattr(entity, k)
            for k in dir(entity)
            if not k.startswith("_") and not callable(getattr(entity, k))
        }

    @classmethod
    def _extract_metadata(cls, entity: dict[str, Any]) -> dict[str, Any]:
        reserved = {"id", "text", "embedding", "distance", "score"}
        return {k: v for k, v in entity.items() if k not in reserved}

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
        self._ensure_collection(vecs.shape[1])
        rows = []
        for text, vector, extra in zip(texts, vecs, meta, strict=True):
            row = {"text": text, "embedding": vector.tolist(), **extra}
            rows.append(row)
        self._client.insert(collection_name=self._collection_name, data=rows)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        if not self._client.has_collection(collection_name=self._collection_name):
            return []

        q_vec = await self._embeddings.embed_one(query)
        results = self._client.search(
            collection_name=self._collection_name,
            data=[q_vec.tolist()],
            limit=top_k,
            filter=self._build_expr(metadata_filter),
            output_fields=["text"],
            search_params=self._build_search_params(),
        )

        hits = results[0] if results and isinstance(results[0], list) else results or []
        out: list[dict] = []
        for hit in hits:
            entity = self._entity_to_dict(getattr(hit, "entity", hit))
            score = getattr(hit, "distance", getattr(hit, "score", 0.0))
            out.append(
                {
                    "text": entity.get("text", ""),
                    "score": float(score),
                    "metadata": self._extract_metadata(entity),
                }
            )
        return out
