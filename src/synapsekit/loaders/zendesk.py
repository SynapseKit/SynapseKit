from __future__ import annotations

import asyncio
from typing import Any

from .base import Document

_VALID_STATUSES = {"open", "pending", "solved", "all"}


class ZendeskLoader:
    """Load Zendesk tickets via the Zendesk REST API.

    Example::

        loader = ZendeskLoader(
            subdomain="mycompany",
            email="agent@mycompany.com",
            api_token="your-api-token",
            status="open",
        )
        docs = loader.load()

    pip install synapsekit[zendesk]  (requires requests>=2.28)
    """

    def __init__(
        self,
        subdomain: str,
        email: str,
        api_token: str,
        status: str = "open",
        limit: int | None = None,
    ) -> None:
        if not subdomain:
            raise ValueError("subdomain must be provided")
        if not email:
            raise ValueError("email must be provided")
        if not api_token:
            raise ValueError("api_token must be provided")
        if status not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {_VALID_STATUSES!r}, got {status!r}")

        self._subdomain = subdomain
        self._email = email
        self._api_token = api_token
        self._status = status
        self._limit = limit

    def load(self) -> list[Document]:
        try:
            import requests
        except ImportError:
            raise ImportError("requests required: pip install synapsekit[zendesk]") from None

        auth = (f"{self._email}/token", self._api_token)
        base_url = f"https://{self._subdomain}.zendesk.com/api/v2/tickets.json"
        params: dict[str, Any] = {}
        if self._status != "all":
            params["status"] = self._status

        docs: list[Document] = []
        url: str | None = base_url

        while url:
            response = requests.get(url, auth=auth, params=params)
            response.raise_for_status()
            data = response.json()
            tickets: list[dict[str, Any]] = data.get("tickets", [])

            for ticket in tickets:
                subject = ticket.get("subject", "")
                description = ticket.get("description", "")
                text = subject + ("\n" + description if description else "")
                metadata: dict[str, Any] = {
                    "source": "zendesk",
                    "ticket_id": ticket.get("id"),
                    "status": ticket.get("status"),
                    "priority": ticket.get("priority"),
                    "requester_id": ticket.get("requester_id"),
                    "created_at": ticket.get("created_at"),
                }
                docs.append(Document(text=text, metadata=metadata))

                if self._limit is not None and len(docs) >= self._limit:
                    return docs

            url = data.get("next_page")
            params = {}  # next_page URL already contains params

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
