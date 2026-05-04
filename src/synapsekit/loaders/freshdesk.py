from __future__ import annotations

import asyncio
from typing import Any

from .base import Document


class FreshdeskLoader:
    """Load Freshdesk tickets via the Freshdesk REST API.

    Status codes: 2=open, 3=pending, 4=resolved, 5=closed.

    Example::

        loader = FreshdeskLoader(
            domain="mycompany",
            api_key="your-api-key",
            status=2,
        )
        docs = loader.load()

    pip install synapsekit[freshdesk]  (requires requests>=2.28)
    """

    def __init__(
        self,
        domain: str,
        api_key: str,
        status: int = 2,
        limit: int | None = None,
    ) -> None:
        if not domain:
            raise ValueError("domain must be provided")
        if not api_key:
            raise ValueError("api_key must be provided")

        self._domain = domain
        self._api_key = api_key
        self._status = status
        self._limit = limit

    def load(self) -> list[Document]:
        try:
            import requests
        except ImportError:
            raise ImportError("requests required: pip install synapsekit[freshdesk]") from None

        # Freshdesk uses api_key as username and any string as password
        auth = (self._api_key, "X")
        base_url = f"https://{self._domain}.freshdesk.com/api/v2/tickets"

        docs: list[Document] = []
        page = 1

        while True:
            params: dict[str, Any] = {"status": self._status, "page": page, "per_page": 100}
            response = requests.get(base_url, auth=auth, params=params)
            response.raise_for_status()
            tickets: list[dict[str, Any]] = response.json()

            if not tickets:
                break

            for ticket in tickets:
                subject = ticket.get("subject", "")
                description = ticket.get("description_text", "")
                text = subject + ("\n" + description if description else "")
                metadata: dict[str, Any] = {
                    "source": "freshdesk",
                    "ticket_id": ticket.get("id"),
                    "status": ticket.get("status"),
                    "priority": ticket.get("priority"),
                    "requester_id": ticket.get("requester_id"),
                    "created_at": ticket.get("created_at"),
                }
                docs.append(Document(text=text, metadata=metadata))

                if self._limit is not None and len(docs) >= self._limit:
                    return docs

            page += 1

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
