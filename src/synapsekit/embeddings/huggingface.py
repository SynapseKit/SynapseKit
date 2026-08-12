"""Hugging Face embeddings provider (Inference API / TEI)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import BaseEmbeddings


class HuggingFaceEmbeddings(BaseEmbeddings):
    """Async embeddings backed by Hugging Face.

    Two endpoints are supported:

    - **Inference API** (default): ``POST https://api-inference.huggingface.co/
      models/{model}`` with ``{"inputs": texts}`` — returns a JSON list of
      vectors. Default model ``BAAI/bge-base-en-v1.5`` (768 dims).
    - **TEI** (Text Embeddings Inference, self-hosted): pass ``base_url``,
      e.g. ``base_url="http://localhost:8080"`` — posts ``{"inputs": texts}``
      to ``{base_url}/embed``.

    Usage::

        emb = HuggingFaceEmbeddings()                        # BAAI/bge-base-en-v1.5
        emb = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
        emb = HuggingFaceEmbeddings(base_url="http://localhost:8080")
        vecs = await emb.embed(["hello", "world"])

    Requires ``httpx``: ``pip install synapsekit[huggingface]``
    """

    dimensions: int | None = 768

    def __init__(
        self,
        model: str = "BAAI/bge-base-en-v1.5",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(batch_size=batch_size, normalize=normalize)
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") if base_url else None
        self._timeout = timeout
        self._client: Any = None

    def _get_key(self) -> str:
        key = self._api_key or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        if not key:
            raise ValueError("HF_TOKEN is not set")
        return key

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError("httpx required: pip install synapsekit[huggingface]") from None
            self._get_key()  # fail fast on a missing key
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def _url(self) -> str:
        if self._base_url is not None:
            return f"{self._base_url}/embed"
        return f"https://api-inference.huggingface.co/models/{self.model}"

    async def _embed_raw(self, texts: list[str]) -> np.ndarray:
        import asyncio
        import json

        client = self._get_client()
        url = self._url()

        def _request() -> Any:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {self._get_key()}"},
                json={"inputs": texts},
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Hugging Face embedding request failed: HTTP {resp.status_code} "
                    f"{resp.text[:200]}"
                )
            return json.loads(resp.content)

        try:
            data = await asyncio.to_thread(_request)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Hugging Face embedding request failed: {exc}") from exc

        if isinstance(data, dict) and isinstance(data.get("data"), list):
            # Some TEI versions return OpenAI-style objects.
            ordered = sorted(data["data"], key=lambda item: item["index"])
            return np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
        if isinstance(data, list):
            return np.asarray(data, dtype=np.float32)
        raise RuntimeError(f"Unexpected Hugging Face embedding response: {data!r}")
