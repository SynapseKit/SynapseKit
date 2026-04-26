from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

from .base import Document

_VALID_SORTS = {"hot", "new", "top", "rising"}


class RedditLoader:
    """Load Reddit posts from a subreddit.

    If ``client_id`` and ``client_secret`` are provided, OAuth is used.
    Otherwise, the public JSON API is used (no credentials required).

    Example::

        loader = RedditLoader(subreddit="python", sort="hot", limit=10)
        docs = loader.load()
    """

    def __init__(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 10,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str = "synapsekit/1.0",
    ) -> None:
        if not subreddit:
            raise ValueError("subreddit must be provided")
        if sort not in _VALID_SORTS:
            raise ValueError(f"sort must be one of {_VALID_SORTS!r}, got {sort!r}")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        self._subreddit = subreddit
        self._sort = sort
        self._limit = limit
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent

    def load(self) -> list[Document]:
        if self._client_id and self._client_secret:
            return self._load_oauth()
        return self._load_public()

    async def aload(self) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    def _load_public(self) -> list[Document]:
        url = f"https://www.reddit.com/r/{self._subreddit}/{self._sort}.json?limit={self._limit}"
        req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        with urllib.request.urlopen(req) as resp:
            data: dict[str, Any] = json.loads(resp.read().decode())
        posts: list[dict[str, Any]] = data.get("data", {}).get("children", [])
        return self._posts_to_docs(posts)

    def _load_oauth(self) -> list[Document]:
        import base64
        import urllib.parse

        # Obtain bearer token
        credentials = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
        token_req = urllib.request.Request(
            "https://www.reddit.com/api/v1/access_token",
            data=urllib.parse.urlencode({"grant_type": "client_credentials"}).encode(),
            headers={
                "Authorization": f"Basic {credentials}",
                "User-Agent": self._user_agent,
            },
            method="POST",
        )
        with urllib.request.urlopen(token_req) as resp:
            token_data: dict[str, Any] = json.loads(resp.read().decode())
        access_token = token_data["access_token"]

        url = f"https://oauth.reddit.com/r/{self._subreddit}/{self._sort}?limit={self._limit}"
        api_req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "User-Agent": self._user_agent,
            },
        )
        with urllib.request.urlopen(api_req) as resp:
            data = json.loads(resp.read().decode())
        posts = data.get("data", {}).get("children", [])
        return self._posts_to_docs(posts)

    def _posts_to_docs(self, posts: list[dict[str, Any]]) -> list[Document]:
        docs: list[Document] = []
        for child in posts[: self._limit]:
            post: dict[str, Any] = child.get("data", {})
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            text = title + ("\n" + selftext if selftext else "")
            metadata: dict[str, Any] = {
                "source": "reddit",
                "post_id": post.get("id"),
                "subreddit": post.get("subreddit"),
                "score": post.get("score"),
                "url": post.get("url"),
                "author": post.get("author"),
                "created_utc": post.get("created_utc"),
            }
            docs.append(Document(text=text, metadata=metadata))
        return docs
