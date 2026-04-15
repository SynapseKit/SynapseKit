from __future__ import annotations

import asyncio
from typing import Any

from .base import Document


class ElasticsearchLoader:
    """Load documents from an Elasticsearch index."""

    def __init__(
        self,
        url: str,
        index: str,
        query: dict[str, Any] | None = None,
        text_fields: list[str] | None = None,
        limit: int | None = None,
    ) -> None:
        if not url:
            raise ValueError("url must be provided")
        if not index:
            raise ValueError("index must be provided")

        self._url = url
        self._index = index
        self._query = query
        self._text_fields = text_fields or ["content"]
        self._limit = limit

    def load(self) -> list[Document]:
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(
                "elasticsearch required: pip install synapsekit[elasticsearch]"
            ) from None

        client = Elasticsearch(self._url)
        resp = client.search(
            index=self._index,
            query=self._query or {"match_all": {}},
            size=self._limit or 100,
        )

        hits = resp.get("hits", {}).get("hits", [])

        docs: list[Document] = []
        for hit in hits:
            source = hit.get("_source", {})

            text = " ".join(
                str(source[field])
                for field in self._text_fields
                if source.get(field) is not None and source.get(field) != ""
            )

            if not text:
                continue

            metadata: dict[str, Any] = {
                "source": "elasticsearch",
                "index": self._index,
                "id": hit.get("_id"),
            }

            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
