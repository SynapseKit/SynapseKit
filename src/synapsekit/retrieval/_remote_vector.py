"""Shared behaviour for remote vector-store adapters."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

DocumentRecord = tuple[str, dict[str, Any]]


class RemoteVectorStoreSupport:
    """Common contract support for SDK-backed vector stores.

    Remote services own the durable index. The small local document cache is
    used for portable ``save``/``load`` snapshots and for adapters, such as
    Vertex Vector Search, whose nearest-neighbour response contains IDs only.
    """

    def _init_remote_state(self) -> None:
        self._documents: list[DocumentRecord] = []
        self._pending_documents: list[DocumentRecord] = []

    @staticmethod
    def _validate_documents(
        texts: list[str], metadata: list[dict[str, Any]] | None
    ) -> list[DocumentRecord]:
        if metadata is None:
            metadata = [{} for _ in texts]
        if len(metadata) != len(texts):
            raise ValueError("metadata must match texts length")
        return list(zip(texts, metadata, strict=True))

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        if top_k <= 0:
            return 0
        return top_k

    @staticmethod
    def _as_float_list(vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]

    @classmethod
    def _as_float_lists(cls, vectors: Any) -> list[list[float]]:
        return [cls._as_float_list(vector) for vector in vectors]

    def _remember_documents(self, documents: list[DocumentRecord]) -> None:
        self._documents.extend(documents)

    async def _flush_pending(self) -> None:
        pending = list(getattr(self, "_pending_documents", []))
        if not pending:
            return
        self._pending_documents = []
        add = cast(
            Callable[[list[str], list[dict[str, Any]]], Awaitable[None]],
            self.add,  # type: ignore[attr-defined]
        )
        try:
            await add(
                [text for text, _ in pending],
                [metadata for _, metadata in pending],
            )
        except BaseException:
            self._pending_documents = pending + getattr(self, "_pending_documents", [])
            raise

    async def search_mmr(
        self,
        query: str,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run MMR over a provider's nearest-neighbour candidates."""
        top_k = self._validate_top_k(top_k)
        fetch_k = self._validate_top_k(fetch_k)
        if top_k == 0 or fetch_k == 0:
            return []
        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")

        search = cast(
            Callable[..., Awaitable[list[dict[str, Any]]]],
            self.search,  # type: ignore[attr-defined]
        )
        candidates: list[dict[str, Any]] = await search(
            query,
            top_k=max(top_k, fetch_k),
            metadata_filter=metadata_filter,
        )
        if len(candidates) <= top_k:
            return candidates

        embeddings = getattr(self, "_embeddings", None)
        if embeddings is None:
            raise ValueError("search_mmr requires an embedding_backend")
        query_vector = self._as_float_list(await embeddings.embed_one(query))
        document_vectors = self._as_float_lists(
            await embeddings.embed([item.get("text", "") for item in candidates])
        )

        def cosine(left: list[float], right: list[float]) -> float:
            numerator = sum(a * b for a, b in zip(left, right, strict=True))
            left_norm = math.sqrt(sum(value * value for value in left))
            right_norm = math.sqrt(sum(value * value for value in right))
            if left_norm == 0 or right_norm == 0:
                return 0.0
            return numerator / (left_norm * right_norm)

        relevance = [cosine(query_vector, vector) for vector in document_vectors]
        similarities = [
            [cosine(left, right) for right in document_vectors] for left in document_vectors
        ]
        selected: list[int] = []
        remaining = set(range(len(candidates)))
        while remaining and len(selected) < top_k:
            best = max(
                remaining,
                key=lambda index: (
                    lambda_mult * relevance[index]
                    - (1 - lambda_mult)
                    * max((similarities[index][other] for other in selected), default=0.0),
                    relevance[index],
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return [candidates[index] for index in selected]

    def save(self, path: str) -> None:
        """Save locally-added documents as a portable JSON snapshot."""
        documents = [*self._documents, *getattr(self, "_pending_documents", [])]
        payload = {
            "version": 1,
            "documents": [{"text": text, "metadata": metadata} for text, metadata in documents],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> None:
        """Queue a snapshot for insertion into the remote index on next I/O."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported vector-store snapshot")
        raw_documents = payload.get("documents")
        if not isinstance(raw_documents, list):
            raise ValueError("snapshot documents must be a list")
        documents: list[DocumentRecord] = []
        for item in raw_documents:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError("snapshot document must contain text")
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("snapshot metadata must be an object")
            documents.append((item["text"], metadata))
        self._pending_documents = documents
