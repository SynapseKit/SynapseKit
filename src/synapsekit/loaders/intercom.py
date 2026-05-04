from __future__ import annotations

import asyncio
from typing import Any

from .base import Document

_VALID_STATES = {"open", "closed", "all"}


class IntercomLoader:
    """Load Intercom conversations via the Intercom API.

    Example::

        loader = IntercomLoader(
            access_token="your-access-token",
            state="open",
            limit=50,
        )
        docs = loader.load()

    pip install synapsekit[intercom]  (requires requests>=2.28)
    """

    def __init__(
        self,
        access_token: str,
        state: str = "open",
        limit: int | None = None,
    ) -> None:
        if not access_token:
            raise ValueError("access_token must be provided")
        if state not in _VALID_STATES:
            raise ValueError(f"state must be one of {_VALID_STATES!r}, got {state!r}")

        self._access_token = access_token
        self._state = state
        self._limit = limit

    def load(self) -> list[Document]:
        try:
            import requests
        except ImportError:
            raise ImportError("requests required: pip install synapsekit[intercom]") from None

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }
        params: dict[str, Any] = {}
        if self._state != "all":
            params["state"] = self._state

        docs: list[Document] = []
        url: str | None = "https://api.intercom.io/conversations"

        while url:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            conversations: list[dict[str, Any]] = data.get("conversations", [])

            for conv in conversations:
                subject = conv.get("title") or ""
                # First message body lives in source.body
                source: dict[str, Any] = conv.get("source") or {}
                body = source.get("body") or ""
                text = subject + ("\n" + body if body else "")

                # contact is under contacts.contacts[0]
                contacts_wrapper: dict[str, Any] = conv.get("contacts") or {}
                contacts_list: list[dict[str, Any]] = contacts_wrapper.get("contacts") or []
                contact_id = contacts_list[0].get("id") if contacts_list else None

                metadata: dict[str, Any] = {
                    "source": "intercom",
                    "conversation_id": conv.get("id"),
                    "state": conv.get("state"),
                    "created_at": conv.get("created_at"),
                    "contact_id": contact_id,
                }
                docs.append(Document(text=text, metadata=metadata))

                if self._limit is not None and len(docs) >= self._limit:
                    return docs

            pages: dict[str, Any] = data.get("pages") or {}
            next_page: dict[str, Any] | None = pages.get("next")
            url = next_page.get("url") if next_page else None
            params = {}

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)
