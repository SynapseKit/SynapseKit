"""Azure AI Search vector-store adapter."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from ._remote_vector import RemoteVectorStoreSupport
from .base import VectorStore


class AzureAISearchVectorStore(RemoteVectorStoreSupport, VectorStore):
    """Async adapter for Azure AI Search vector and hybrid indexes.

    The adapter creates a vector-capable index on the first write when an
    ``index_client`` is available. Metadata is kept as JSON so arbitrary
    SynapseKit metadata remains lossless across the Azure document API.
    """

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        endpoint: str | None = None,
        api_key: str | None = None,
        index_name: str = "synapsekit-docs",
        credential: Any | None = None,
        search_client: Any | None = None,
        index_client: Any | None = None,
    ) -> None:
        if search_client is None or index_client is None:
            try:
                from azure.core.credentials import AzureKeyCredential
                from azure.search.documents import SearchClient
                from azure.search.documents.indexes import SearchIndexClient
            except ImportError:
                raise ImportError(
                    "azure-search-documents required: pip install synapsekit[azure-ai-search]"
                ) from None
            if credential is None:
                if not endpoint or not api_key:
                    raise ValueError("endpoint and api_key are required")
                credential = AzureKeyCredential(api_key)
            if not endpoint:
                raise ValueError("endpoint is required")
            if index_client is None:
                index_client = SearchIndexClient(endpoint, credential)
            if search_client is None:
                search_client = SearchClient(endpoint, index_name, credential)

        self._embeddings = embedding_backend
        self._endpoint = endpoint
        self._index_name = index_name
        self._search_client = search_client
        self._index_client = index_client
        self._index_created = False
        self._dim: int | None = None
        self._init_remote_state()

    def _ensure_index(self, dim: int) -> None:
        if self._index_created and self._dim == dim:
            return
        if self._index_client is None:
            self._index_created = True
            self._dim = dim
            return
        try:
            self._index_client.get_index(self._index_name)
        except Exception:
            from azure.search.documents.indexes.models import (
                HnswAlgorithmConfiguration,
                SearchableField,
                SearchField,
                SearchFieldDataType,
                SearchIndex,
                SimpleField,
                VectorSearch,
                VectorSearchProfile,
            )

            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SearchableField(name="text", type=SearchFieldDataType.String),
                SearchField(
                    name="metadata",
                    type=SearchFieldDataType.String,
                    searchable=False,
                    filterable=False,
                ),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=dim,
                    vector_search_profile_name="synapsekit-hnsw",
                ),
            ]
            vector_search = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="synapsekit-hnsw-algorithm")],
                profiles=[
                    VectorSearchProfile(
                        name="synapsekit-hnsw",
                        algorithm_configuration_name="synapsekit-hnsw-algorithm",
                    )
                ],
            )
            self._index_client.create_or_update_index(
                SearchIndex(
                    name=self._index_name,
                    fields=fields,
                    vector_search=vector_search,
                )
            )
        self._index_created = True
        self._dim = dim

    def _add_sync(
        self, documents: list[tuple[str, dict[str, Any]]], vectors: list[list[float]]
    ) -> None:
        self._ensure_index(len(vectors[0]))
        payload = [
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "metadata": json.dumps(metadata, ensure_ascii=False),
                "embedding": vector,
            }
            for (text, metadata), vector in zip(documents, vectors, strict=True)
        ]
        self._search_client.upload_documents(payload)

    @staticmethod
    def _parse_document(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            text = item.get("text", "")
            raw_metadata = item.get("metadata", "{}")
            score = item.get("@search.score", item.get("score", 0.0))
        else:
            text = getattr(item, "text", "")
            raw_metadata = getattr(item, "metadata", "{}")
            score = getattr(item, "@search.score", getattr(item, "score", 0.0))
        try:
            metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return {
            "text": text,
            "score": float(score or 0.0),
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

    def _search_sync(
        self,
        query_text: str,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        from azure.search.documents.models import VectorizedQuery

        limit = max(top_k * 10, 100) if metadata_filter else top_k
        query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=limit,
            fields="embedding",
        )
        try:
            rows = self._search_client.search(
                search_text=query_text,
                vector_queries=[query],
                top=limit,
                select=["text", "metadata"],
            )
        except TypeError:
            rows = self._search_client.search(
                search_text=query_text,
                vectors=[query],
                top=limit,
                select=["text", "metadata"],
            )
        results = []
        for row in rows:
            item = self._parse_document(row)
            if metadata_filter and not all(
                item["metadata"].get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            results.append(item)
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
        try:
            return await asyncio.to_thread(self._search_sync, query, vector, top_k, metadata_filter)
        except Exception as exc:
            if exc.__class__.__name__ in {"ResourceNotFoundError", "HttpResponseError"}:
                return []
            raise
