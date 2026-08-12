"""Shared HTTP transport for OpenAI-compatible embeddings providers.

Used by Voyage, Jina, Nomic, mixedbread, and Hugging Face (Inference API /
TEI). All of them accept ``POST <base_url>/embeddings`` with an
``{"model": ..., "input": [...]}`` body and return
``{"data": [{"index": ..., "embedding": [...]}]}``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import BaseEmbeddings


class HTTPEmbeddings(BaseEmbeddings):
    """Async embeddings over a generic ``/embeddings`` HTTP endpoint.

    Constructor arguments:

    - ``model``: model name sent in the request body.
    - ``api_key``: explicit API key; falls back to ``os.environ[env_key]``.
      When neither is available, a ``ValueError`` is raised on first use.
    - ``base_url``: provider base URL; the endpoint is ``{base_url}/embeddings``.
    - ``env_key``: environment variable holding the API key.
    - ``batch_size`` / ``normalize``: inherited from ``BaseEmbeddings``.
    - ``request_extra``: extra static JSON fields merged into every request
      body (e.g. ``task=...`` for Jina, ``task_type=...`` for Nomic).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str,
        env_key: str,
        batch_size: int = 64,
        normalize: bool = True,
        timeout: float = 60.0,
        **request_extra: Any,
    ) -> None:
        super().__init__(batch_size=batch_size, normalize=normalize)
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._env_key = env_key
        self._timeout = timeout
        self._request_extra = request_extra
        self._client: Any = None

    def _get_key(self) -> str:
        key = self._api_key or os.environ.get(self._env_key)
        if not key:
            raise ValueError(f"{self._env_key} is not set")
        return key

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError("httpx required: pip install synapsekit[web]") from None
            self._get_key()  # fail fast on a missing key
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    async def _embed_raw(self, texts: list[str]) -> np.ndarray:
        """POST to ``{base_url}/embeddings`` and parse the response."""
        import asyncio
        import json

        client = self._get_client()
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        payload.update(self._request_extra)

        def _request() -> dict[str, Any]:
            resp = client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._get_key()}"},
                json=payload,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Embedding request failed: HTTP {resp.status_code} {resp.text[:200]}"
                )
            return json.loads(resp.content)

        try:
            data = await asyncio.to_thread(_request)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc

        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected embedding response: {data!r}")

        ordered = sorted(items, key=lambda item: item["index"])
        vecs = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
        return vecs
