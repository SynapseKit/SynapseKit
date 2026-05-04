from __future__ import annotations

import asyncio
from typing import Any

from .base import Document


class TrelloLoader:
    """Load Trello cards from a board.

    Example::

        loader = TrelloLoader(
            api_key="your-api-key",
            token="your-token",
            board_id="your-board-id",
        )
        docs = loader.load()

    pip install synapsekit[trello]  (requires requests>=2.28)
    """

    def __init__(
        self,
        api_key: str,
        token: str,
        board_id: str,
        list_names: list[str] | None = None,
        include_attachments: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be provided")
        if not token:
            raise ValueError("token must be provided")
        if not board_id:
            raise ValueError("board_id must be provided")

        self._api_key = api_key
        self._token = token
        self._board_id = board_id
        self._list_names = [n.lower() for n in list_names] if list_names else None
        self._include_attachments = include_attachments

    def load(self) -> list[Document]:
        try:
            import requests
        except ImportError:
            raise ImportError("requests required: pip install synapsekit[trello]") from None

        list_map = self._fetch_lists(requests)

        params: dict[str, Any] = {
            "key": self._api_key,
            "token": self._token,
            "fields": "name,desc,idList,labels,shortUrl",
            "attachments": "true" if self._include_attachments else "false",
        }
        url = f"https://api.trello.com/1/boards/{self._board_id}/cards"
        response = requests.get(url, params=params)
        response.raise_for_status()
        cards: list[dict[str, Any]] = response.json()

        docs: list[Document] = []
        for card in cards:
            list_id = card.get("idList", "")
            list_name = list_map.get(list_id, "")

            if self._list_names is not None and list_name.lower() not in self._list_names:
                continue

            name = card.get("name", "")
            desc = card.get("desc", "")
            text = name + ("\n" + desc if desc else "")

            labels = [lbl.get("name", "") for lbl in card.get("labels", [])]
            metadata: dict[str, Any] = {
                "source": "trello",
                "board_id": self._board_id,
                "card_id": card.get("id", ""),
                "list_name": list_name,
                "labels": labels,
                "url": card.get("shortUrl", ""),
            }
            docs.append(Document(text=text, metadata=metadata))

        return docs

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _fetch_lists(self, requests: Any) -> dict[str, str]:
        url = f"https://api.trello.com/1/boards/{self._board_id}/lists"
        params = {"key": self._api_key, "token": self._token, "fields": "name"}
        response = requests.get(url, params=params)
        response.raise_for_status()
        lists: list[dict[str, Any]] = response.json()
        return {lst["id"]: lst["name"] for lst in lists}
