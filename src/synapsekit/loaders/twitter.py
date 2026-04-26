from __future__ import annotations

import asyncio
from typing import Any

from .base import Document


class TwitterLoader:
    """Load tweets from a search query or user timeline via Twitter API v2.

    Either ``query`` or ``username`` must be provided.

    Example::

        loader = TwitterLoader(
            bearer_token="your-bearer-token",
            query="python programming",
            max_results=10,
        )
        docs = loader.load()

    pip install synapsekit[twitter]  (requires requests>=2.28)
    """

    def __init__(
        self,
        bearer_token: str,
        query: str | None = None,
        username: str | None = None,
        max_results: int = 10,
    ) -> None:
        if not bearer_token:
            raise ValueError("bearer_token must be provided")
        if not query and not username:
            raise ValueError("Either query or username must be provided")
        if max_results < 10 or max_results > 100:
            raise ValueError("max_results must be between 10 and 100")

        self._bearer_token = bearer_token
        self._query = query
        self._username = username
        self._max_results = max_results

    def load(self) -> list[Document]:
        try:
            import requests
        except ImportError:
            raise ImportError("requests required: pip install synapsekit[twitter]") from None

        headers = {"Authorization": f"Bearer {self._bearer_token}"}

        if self._query:
            return self._load_search(requests, headers)
        return self._load_timeline(requests, headers)

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _load_search(self, requests: Any, headers: dict[str, str]) -> list[Document]:
        params: dict[str, Any] = {
            "query": self._query,
            "max_results": self._max_results,
            "tweet.fields": "author_id,created_at,public_metrics",
        }
        response = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        return self._tweets_to_docs(data.get("data", []))

    def _load_timeline(self, requests: Any, headers: dict[str, str]) -> list[Document]:
        # Resolve username to user ID
        user_resp = requests.get(
            f"https://api.twitter.com/2/users/by/username/{self._username}",
            headers=headers,
        )
        user_resp.raise_for_status()
        user_data = user_resp.json()
        user_id = user_data.get("data", {}).get("id")

        params: dict[str, Any] = {
            "max_results": self._max_results,
            "tweet.fields": "author_id,created_at,public_metrics",
        }
        response = requests.get(
            f"https://api.twitter.com/2/users/{user_id}/tweets",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        return self._tweets_to_docs(data.get("data", []))

    def _tweets_to_docs(self, tweets: list[dict[str, Any]]) -> list[Document]:
        docs: list[Document] = []
        for tweet in tweets:
            text = tweet.get("text", "")
            metadata: dict[str, Any] = {
                "source": "twitter",
                "tweet_id": tweet.get("id"),
                "author_id": tweet.get("author_id"),
                "created_at": tweet.get("created_at"),
                "public_metrics": tweet.get("public_metrics"),
            }
            docs.append(Document(text=text, metadata=metadata))
        return docs
