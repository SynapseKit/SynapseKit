from __future__ import annotations

import asyncio
from typing import Any

from .base import Document


class FirestoreLoader:
    """Load documents from a Firestore collection.

    Example::

        loader = FirestoreLoader(
            project_id="my-project",
            collection="articles",
            credentials_path="/path/to/service-account.json",
        )
        docs = loader.load()

    pip install synapsekit[firestore]  (requires google-cloud-firestore>=2.13)
    """

    def __init__(
        self,
        project_id: str,
        collection: str,
        credentials_path: str | None = None,
        limit: int | None = None,
        metadata_fields: list[str] | None = None,
    ) -> None:
        if not project_id:
            raise ValueError("project_id must be provided")
        if not collection:
            raise ValueError("collection must be provided")

        self._project_id = project_id
        self._collection = collection
        self._credentials_path = credentials_path
        self._limit = limit
        self._metadata_fields = metadata_fields

    def load(self) -> list[Document]:
        try:
            from google.cloud import firestore
        except ImportError:
            raise ImportError(
                "google-cloud-firestore required: pip install synapsekit[firestore]"
            ) from None

        if self._credentials_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                self._credentials_path
            )
            client = firestore.Client(project=self._project_id, credentials=credentials)
        else:
            client = firestore.Client(project=self._project_id)

        query: Any = client.collection(self._collection)
        if self._limit is not None:
            query = query.limit(self._limit)

        snapshot = query.stream()

        metadata_set = set(self._metadata_fields) if self._metadata_fields else None

        docs: list[Document] = []
        for doc_ref in snapshot:
            data: dict[str, Any] = doc_ref.to_dict() or {}

            if metadata_set:
                text_parts: list[str] = []
                metadata: dict[str, Any] = {
                    "source": "firestore",
                    "collection": self._collection,
                    "doc_id": doc_ref.id,
                }
                for key, value in data.items():
                    if key in metadata_set:
                        metadata[key] = value
                    else:
                        text_parts.append(f"{key}: {value}")
                text = "\n".join(text_parts)
            else:
                text = "\n".join(f"{k}: {v}" for k, v in data.items())
                metadata = {
                    "source": "firestore",
                    "collection": self._collection,
                    "doc_id": doc_ref.id,
                }

            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
