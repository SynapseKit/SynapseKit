"""Deep Lake (Activeloop) vector-store adapter."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from typing import Any

from ..embeddings.backend import SynapsekitEmbeddings
from ._remote_vector import RemoteVectorStoreSupport
from .base import VectorStore


class DeepLakeVectorStore(RemoteVectorStoreSupport, VectorStore):
    """Async adapter for local or cloud Deep Lake datasets."""

    def __init__(
        self,
        embedding_backend: SynapsekitEmbeddings,
        dataset_path: str = "./synapsekit_deeplake",
        token: str | None = None,
        dataset: Any | None = None,
    ) -> None:
        if dataset is None:
            try:
                import deeplake
            except ImportError:
                raise ImportError("deeplake required: pip install synapsekit[deeplake]") from None
            try:
                dataset = deeplake.load(dataset_path, token=token)
            except Exception:
                dataset = deeplake.create(dataset_path, token=token)
        self._embeddings = embedding_backend
        self._dataset_path = dataset_path
        self._dataset = dataset
        self._local_vectors: list[list[float]] = []
        self._init_remote_state()

    def _add_columns(self, dim: int) -> None:
        """Create the minimal schema when the dataset supports dynamic columns."""
        if getattr(self, "_schema_ready", False):
            return
        try:
            import deeplake

            types = deeplake.types
            for name, dtype in (
                ("text", types.Text()),
                ("metadata", types.Dict()),
                ("embedding", types.Embedding(dim)),
            ):
                with contextlib.suppress(Exception):
                    self._dataset.add_column(name, dtype)
        except (ImportError, AttributeError):
            pass
        self._schema_ready = True

    def _add_sync(
        self, documents: list[tuple[str, dict[str, Any]]], vectors: list[list[float]]
    ) -> None:
        self._add_columns(len(vectors[0]))
        rows = [
            {"text": text, "metadata": metadata, "embedding": vector}
            for (text, metadata), vector in zip(documents, vectors, strict=True)
        ]
        self._dataset.append(rows)
        commit = getattr(self._dataset, "commit", None)
        if commit:
            commit()
        self._local_vectors.extend(vectors)

    @staticmethod
    def _result_value(result: Any, key: str, index: int) -> Any:
        if isinstance(result, dict):
            values = result.get(key)
            if values is not None:
                try:
                    return values[index]
                except (IndexError, KeyError, TypeError):
                    return None
        return None

    def _search_dataset(self, vector: list[float], limit: int) -> list[dict[str, Any]] | None:
        search = getattr(self._dataset, "search", None)
        if not callable(search):
            return None
        try:
            response = search(
                embedding_data=vector,
                k=limit,
                return_tensors=["text", "metadata", "embedding"],
            )
        except (AttributeError, TypeError, NotImplementedError):
            return None
        if not isinstance(response, dict):
            return None
        texts = response.get("text", [])
        count = len(texts)
        output = []
        for index in range(count):
            metadata = self._result_value(response, "metadata", index) or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            output.append(
                {
                    "text": self._result_value(response, "text", index) or "",
                    "score": float(self._result_value(response, "score", index) or 0.0),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                }
            )
        return output

    def _search_sync(
        self, vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        limit = max(top_k * 10, 100) if metadata_filter else top_k
        dataset_results = self._search_dataset(vector, limit)
        if dataset_results is not None:
            results = dataset_results
        else:
            results = []
            for (text, metadata), candidate in zip(
                self._documents,
                self._local_vectors,
                strict=False,
            ):
                numerator = sum(a * b for a, b in zip(vector, candidate, strict=True))
                left = math.sqrt(sum(value * value for value in vector))
                right = math.sqrt(sum(value * value for value in candidate))
                score = numerator / (left * right) if left and right else 0.0
                results.append({"text": text, "score": score, "metadata": metadata})
            results.sort(key=lambda item: item["score"], reverse=True)
        output = []
        for item in results:
            if metadata_filter and not all(
                item["metadata"].get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            output.append(item)
            if len(output) == top_k:
                break
        return output

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
