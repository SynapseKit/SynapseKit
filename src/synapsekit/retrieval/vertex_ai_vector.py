"""Vertex AI Vector Search adapter."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from ._remote_vector import RemoteVectorStoreSupport
from .base import VectorStore


class VertexAIVectorStore(RemoteVectorStoreSupport, VectorStore):
    """Async adapter for the Vertex AI Matching Engine APIs.

    Vertex returns nearest-neighbour IDs rather than source documents. The
    adapter keeps the source payload alongside the ID in its local cache. Set
    ``document_store_path`` to persist that mapping across reconnects or
    processes; otherwise source recovery is limited to this instance.
    """

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        project: str | None = None,
        location: str = "us-central1",
        index_endpoint_name: str | None = None,
        deployed_index_id: str | None = None,
        index_name: str | None = None,
        index_endpoint: Any | None = None,
        index: Any | None = None,
        document_store_path: str | None = None,
    ) -> None:
        if index_endpoint is None or index is None:
            try:
                from google.cloud import aiplatform
            except ImportError:
                raise ImportError(
                    "google-cloud-aiplatform required: pip install synapsekit[vertex-vector]"
                ) from None
            aiplatform.init(project=project, location=location)
            if index_endpoint is None:
                if not index_endpoint_name:
                    raise ValueError("index_endpoint_name is required")
                index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                    index_endpoint_name=index_endpoint_name
                )
            if index is None and index_name:
                index = aiplatform.MatchingEngineIndex(index_name=index_name)

        if index_endpoint is None:
            raise ValueError("index_endpoint or index_endpoint_name is required")
        self._embeddings = embedding_backend
        self._endpoint = index_endpoint
        self._index = index
        self._deployed_index_id = deployed_index_id
        self._document_store_path = Path(document_store_path) if document_store_path else None
        self._documents_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
        self._init_remote_state()
        self._load_document_store()

    def _load_document_store(self) -> None:
        if self._document_store_path is None or not self._document_store_path.exists():
            return
        payload = json.loads(self._document_store_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Vertex AI document store must be an object")
        for datapoint_id, item in payload.items():
            if not isinstance(datapoint_id, str) or not isinstance(item, dict):
                raise ValueError("invalid Vertex AI document store entry")
            text = item.get("text")
            metadata = item.get("metadata", {})
            if not isinstance(text, str) or not isinstance(metadata, dict):
                raise ValueError("invalid Vertex AI document store entry")
            self._documents_by_id[datapoint_id] = (text, metadata)

    def _persist_document_store(self) -> None:
        if self._document_store_path is None:
            return
        self._document_store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            datapoint_id: {"text": text, "metadata": metadata}
            for datapoint_id, (text, metadata) in self._documents_by_id.items()
        }
        self._document_store_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _neighbor_value(neighbor: Any, key: str, default: Any = None) -> Any:
        if isinstance(neighbor, dict):
            return neighbor.get(key, default)
        return getattr(neighbor, key, default)

    def _add_sync(
        self, documents: list[tuple[str, dict[str, Any]]], vectors: list[list[float]]
    ) -> None:
        if self._index is None:
            raise ValueError("index is required for Vertex AI Vector Search writes")
        datapoints = []
        document_updates: dict[str, tuple[str, dict[str, Any]]] = {}
        for (text, metadata), vector in zip(documents, vectors, strict=True):
            datapoint_id = str(uuid.uuid4())
            datapoints.append(
                {
                    "datapoint_id": datapoint_id,
                    "feature_vector": vector,
                    "restricts": [
                        {"namespace": key, "allow_list": [str(value)]}
                        for key, value in metadata.items()
                        if isinstance(value, (str, int, float, bool))
                    ],
                }
            )
            document_updates[datapoint_id] = (text, metadata)
        try:
            self._index.upsert_datapoints(datapoints=datapoints)
        except TypeError:
            self._index.upsert_datapoints(datapoints)
        self._documents_by_id.update(document_updates)
        self._persist_document_store()

    def _search_sync(
        self, vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if not self._deployed_index_id:
            raise ValueError("deployed_index_id is required for Vertex AI Vector Search")
        limit = max(top_k * 10, 100) if metadata_filter else top_k
        try:
            groups = self._endpoint.find_neighbors(
                deployed_index_id=self._deployed_index_id,
                queries=[vector],
                num_neighbors=limit,
            )
        except TypeError:
            groups = self._endpoint.find_neighbors(
                self._deployed_index_id,
                [vector],
                num_neighbors=limit,
            )
        neighbors = groups[0] if groups else []
        results = []
        for neighbor in neighbors:
            datapoint_id = self._neighbor_value(
                neighbor, "id", self._neighbor_value(neighbor, "datapoint_id", "")
            )
            document = self._documents_by_id.get(str(datapoint_id))
            if document is None:
                continue
            text, metadata = document
            if metadata_filter and not all(
                metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            score = self._neighbor_value(
                neighbor, "distance", self._neighbor_value(neighbor, "score", 0.0)
            )
            results.append({"text": text, "score": float(score), "metadata": metadata})
            if len(results) == top_k:
                break
        return results

    async def add(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        if not texts:
            return
        await self._flush_pending()
        documents = self._validate_documents(texts, metadata)
        vectors = self._as_float_lists(await self._embeddings.embed(texts))
        await asyncio.to_thread(self._add_sync, documents, vectors)
        self._remember_documents(documents)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        top_k = self._validate_top_k(top_k)
        if top_k == 0:
            return []
        await self._flush_pending()
        vector = self._as_float_list(await self._embeddings.embed_one(query))
        return await asyncio.to_thread(self._search_sync, vector, top_k, metadata_filter)


VertexVectorStore = VertexAIVectorStore
