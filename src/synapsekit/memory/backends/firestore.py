"""Firestore-backed AgentMemory storage."""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any

from ..base import BaseMemoryBackend, MemoryRecord, MemoryType
from ._common import memory_record_document_id
from ._serialization import memory_record_from_payload, memory_record_to_payload


def _agent_id_filter(agent_id: str) -> Any:
    from google.cloud.firestore_v1.base_query import FieldFilter

    return FieldFilter("agent_id", "==", agent_id)


class FirestoreMemoryBackend(BaseMemoryBackend):
    """Persist memory records with Firestore's asynchronous client."""

    def __init__(
        self,
        project_id: str = "synapsekit-local",
        collection: str = "agent_memory",
        *,
        credentials_path: str | None = None,
        database: str = "(default)",
        client: Any | None = None,
    ) -> None:
        if not project_id or not collection:
            raise ValueError("project_id and collection must be provided")
        self._project_id = project_id
        self._collection_name = collection
        self._credentials_path = credentials_path
        self._database = database
        self._client = client
        self._collection: Any | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        async with self._init_lock:
            if self._collection is None:
                await self._initialize()
        return self._collection

    async def _initialize(self) -> None:
        if self._client is None:
            try:
                from google.cloud.firestore_v1 import AsyncClient
            except ImportError:
                raise ImportError(
                    "google-cloud-firestore is required for FirestoreMemoryBackend; "
                    "install it with `pip install synapsekit[firestore]`"
                ) from None
            kwargs: dict[str, Any] = {
                "project": self._project_id,
                "database": self._database,
            }
            if self._credentials_path:
                from google.oauth2 import service_account

                kwargs["credentials"] = service_account.Credentials.from_service_account_file(
                    self._credentials_path
                )
            self._client = AsyncClient(**kwargs)
        self._collection = self._client.collection(self._collection_name)

    @staticmethod
    def _to_document(record: MemoryRecord) -> dict[str, Any]:
        payload = memory_record_to_payload(record)
        return {
            "record_id": record.id,
            "agent_id": record.agent_id,
            "content": record.content,
            "memory_type": record.memory_type,
            "embedding": payload["embedding"],
            "embedding_dimension": payload["embedding_dimension"],
            "created_at": payload["created_at"],
            "accessed_at": payload["accessed_at"],
            "access_count": payload["access_count"],
            "ttl_days": payload["ttl_days"],
            "metadata": payload["metadata"],
        }

    @staticmethod
    def _from_document(document: dict[str, Any]) -> MemoryRecord:
        payload = dict(document)
        payload["id"] = document["record_id"]
        return memory_record_from_payload(payload)

    async def store(self, record: MemoryRecord) -> None:
        collection = await self._ensure_collection()
        await collection.document(memory_record_document_id(record.agent_id, record.id)).set(
            self._to_document(record)
        )

    async def fetch(
        self,
        agent_id: str,
        memory_type: MemoryType | None = None,
        *,
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        collection = await self._ensure_collection()
        records: list[MemoryRecord] = []
        async for snapshot in collection.where(filter=_agent_id_filter(agent_id)).stream():
            record = self._from_document(snapshot.to_dict())
            if memory_type is None or record.memory_type == memory_type:
                records.append(record)
        records.sort(key=lambda record: record.created_at)
        if include_expired:
            return records
        now = datetime.now(timezone.utc)
        return [record for record in records if not record.is_expired(now)]

    async def touch(
        self,
        agent_id: str,
        record_id: str,
        *,
        accessed_at: datetime | None = None,
    ) -> None:
        collection = await self._ensure_collection()
        reference = collection.document(memory_record_document_id(agent_id, record_id))
        snapshot = await reference.get()
        if not snapshot.exists:
            return
        data = snapshot.to_dict()
        if data.get("agent_id") != agent_id:
            return
        timestamp = (accessed_at or datetime.now(timezone.utc)).isoformat()
        await reference.update(
            {"accessed_at": timestamp, "access_count": int(data.get("access_count", 0)) + 1}
        )

    async def delete(self, agent_id: str, record_id: str) -> bool:
        collection = await self._ensure_collection()
        reference = collection.document(memory_record_document_id(agent_id, record_id))
        snapshot = await reference.get()
        if not snapshot.exists or snapshot.to_dict().get("agent_id") != agent_id:
            return False
        await reference.delete()
        return True

    async def clear(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        collection = await self._ensure_collection()
        references: list[Any] = []
        async for snapshot in collection.where(filter=_agent_id_filter(agent_id)).stream():
            data = snapshot.to_dict()
            if memory_type is None or data.get("memory_type") == memory_type:
                references.append(snapshot.reference)
        for reference in references:
            await reference.delete()
        return len(references)

    async def count(self, agent_id: str, memory_type: MemoryType | None = None) -> int:
        collection = await self._ensure_collection()
        count = 0
        async for snapshot in collection.where(filter=_agent_id_filter(agent_id)).stream():
            if memory_type is None or snapshot.to_dict().get("memory_type") == memory_type:
                count += 1
        return count

    async def prune_expired(self, *, now: datetime | None = None) -> int:
        collection = await self._ensure_collection()
        current = now or datetime.now(timezone.utc)
        references: list[Any] = []
        async for snapshot in collection.stream():
            if self._from_document(snapshot.to_dict()).is_expired(current):
                references.append(snapshot.reference)
        for reference in references:
            await reference.delete()
        return len(references)

    async def aclose(self) -> None:
        if self._client is not None:
            result = self._client.close()
            if inspect.isawaitable(result):
                await result
            self._client = None
            self._collection = None
