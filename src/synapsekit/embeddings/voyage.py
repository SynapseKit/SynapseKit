"""Voyage AI embeddings provider (voyage-3)."""

from __future__ import annotations

from .http import HTTPEmbeddings


class VoyageEmbeddings(HTTPEmbeddings):
    """Async embeddings backed by the Voyage AI Embeddings API.

    Voyage models are retrieval-optimized and strong on code/finance domains.

    Usage::

        emb = VoyageEmbeddings(api_key="pa-...")            # or VOYAGE_API_KEY
        vecs = await emb.embed(["hello", "world"])          # (2, 1024) float32

    Requires ``httpx``: ``pip install synapsekit[voyage]``
    """

    dimensions: int | None = 1024

    def __init__(
        self,
        model: str = "voyage-3",
        *,
        api_key: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            model,
            api_key=api_key,
            base_url="https://api.voyageai.com/v1",
            env_key="VOYAGE_API_KEY",
            batch_size=batch_size,
            normalize=normalize,
            timeout=timeout,
        )
