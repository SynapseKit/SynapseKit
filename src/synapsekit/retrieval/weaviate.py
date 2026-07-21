"""WeaviateVectorStore — Weaviate-backed vector store backend (weaviate-client v4)."""

from __future__ import annotations

import asyncio

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class WeaviateVectorStore(VectorStore):
    """Weaviate-backed vector store. Embeds externally via SynapsekitEmbeddings.

    The collection is created on the first ``add()`` (no server-side vectorizer);
    ``search()`` before the collection exists returns ``[]``.
    """

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
            auth_config = weaviate.classes.init.Auth.api_key(api_key) if api_key else None
            host = url.split("://")[1] if "://" in url else url
            secure = url.startswith("https")
            # v4 needs an explicit gRPC endpoint; default to the same host on the
            # standard gRPC port unless the caller injects a preconfigured client.
            self._client = weaviate.connect_to_custom(
                http_host=host,
                http_port=443 if secure else 80,
                http_secure=secure,
                grpc_host=host,
                grpc_port=443 if secure else 50051,
                grpc_secure=secure,
                auth_credentials=auth_config,
            )
        else:
            self._client = weaviate.connect_to_local()

    def _ensure_collection(self):  # type: ignore[no-untyped-def]
        import weaviate.classes.config as wvc

        if not self._client.collections.exists(self._collection_name):
            self._client.collections.create(
                name=self._collection_name,
                vectorizer_config=wvc.Configure.Vectorizer.none(),
            )
        return self._client.collections.get(self._collection_name)

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return
        from weaviate.classes.data import DataObject

        meta = metadata or [{} for _ in texts]
        if len(meta) != len(texts):
            raise ValueError("metadata must match texts length")

        vecs = await self._embeddings.embed(texts)

        def _write() -> None:
            # Blocking weaviate client — run off the event loop to keep async alive.
            collection = self._ensure_collection()
            objects = [
                DataObject(
                    properties={"content": texts[i], **dict(meta[i])},
                    vector=vecs[i].tolist(),
                )
                for i in range(len(texts))
            ]
            collection.data.insert_many(objects)

        await asyncio.to_thread(_write)

    async def search(
        self, query: str, top_k: int = 5, metadata_filter: dict | None = None
    ) -> list[dict]:
        from weaviate.classes.query import Filter, MetadataQuery

        q_vec = await self._embeddings.embed_one(query)

        wfilter = None
        if metadata_filter:
            conditions = [
                Filter.by_property(key).equal(value) for key, value in metadata_filter.items()
            ]
            wfilter = conditions[0] if len(conditions) == 1 else Filter.all_of(conditions)

        def _query() -> list[dict]:
            if not self._client.collections.exists(self._collection_name):
                return []
            collection = self._client.collections.get(self._collection_name)
            response = collection.query.near_vector(
                near_vector=q_vec.tolist(),
                limit=top_k,
                filters=wfilter,
                return_metadata=MetadataQuery(distance=True),
            )
            out: list[dict] = []
            for obj in response.objects:
                props = dict(obj.properties)
                text = props.pop("content", "")
                distance = obj.metadata.distance if obj.metadata else None
                # Weaviate returns cosine distance (0 = identical); expose a
                # higher-is-better similarity, results already ranked nearest-first.
                score = (1.0 - distance) if distance is not None else 0.0
                out.append({"text": text, "score": score, "metadata": props})
            return out

        return await asyncio.to_thread(_query)
