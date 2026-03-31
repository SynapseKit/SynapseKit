from __future__ import annotations

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class WeaviateVectorStore(VectorStore):
    """Weaviate-backed vector store. Embeds externally via SynapsekitEmbeddings."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        collection_name: str = "SynapseKit",
        client: object | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "weaviate-client required: pip install synapsekit[weaviate]"
            ) from None

        self._embeddings = embedding_backend
        self._collection_name = collection_name

        if client is not None:
            self._client = client
        elif url is not None:
            auth_config = weaviate.AuthApiKey(api_key=api_key) if api_key else None
            self._client = weaviate.connect_to_custom(
                http_host=url.split("://")[1] if "://" in url else url,
                http_port=443 if url.startswith("https") else 80,
                http_secure=url.startswith("https"),
                auth_credentials=auth_config,
            )
        else:
            self._client = weaviate.connect_to_local()

        self._collection = self._client.collections.get(collection_name)

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return
        meta = metadata or [{} for _ in texts]
        vecs = await self._embeddings.embed(texts)

        objects = [
            {
                "content": text,
                **{k: v for k, v in m.items() if k != "content"},
            }
            for text, m in zip(texts, meta, strict=True)
        ]

        data_objects = [{k: v for k, v in obj.items() if k != "vector"} for obj in objects]

        self._collection.data.insert_many(
            data_objects=data_objects,
            vectors=vecs.tolist(),
        )

    async def search(
        self, query: str, top_k: int = 5, metadata_filter: dict | None = None
    ) -> list[dict]:
        count = self._collection.query.fetch.total
        if count == 0:
            return []
        q_vec = await self._embeddings.embed_one(query)

        results = self._collection.search(
            search_vector=q_vec.tolist(),
            limit=top_k,
            return_metadata=["distance", "score"],
            return_properties=["content"],
        )

        out = []
        for obj in results.objects:
            item = {
                "text": obj.properties.get("content", ""),
                "score": obj.metadata.score if obj.metadata else 0.0,
                "metadata": {k: v for k, v in obj.properties.items() if k != "content"},
            }
            out.append(item)

        if metadata_filter:
            out = [
                item
                for item in out
                if all(item["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        return out

    def save(self, path: str) -> None:
        if hasattr(self._client, "backup"):
            self._client.backup.create(
                backup_id=f"synapsekit_{self._collection_name}",
                collection_names=[self._collection_name],
                backend="filesystem",
            )
        raise NotImplementedError("Save not fully implemented for WeaviateVectorStore")

    def load(self, path: str) -> None:
        raise NotImplementedError("Load not implemented for WeaviateVectorStore")
