"""mixedbread embeddings provider (mxbai-embed-large)."""

from __future__ import annotations

from .http import HTTPEmbeddings


class MixedbreadEmbeddings(HTTPEmbeddings):
    """Async embeddings backed by the mixedbread.ai Embeddings API.

    Usage::

        emb = MixedbreadEmbeddings(api_key="mxb_...")       # or MXBAI_API_KEY
        vecs = await emb.embed(["hello", "world"])          # (2, 1024) float32

    Requires ``httpx``: ``pip install synapsekit[mixedbread]``
    """

    dimensions: int | None = 1024

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        *,
        api_key: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(
            model,
            api_key=api_key,
            base_url="https://api.mixedbread.ai/v1",
            env_key="MXBAI_API_KEY",
            batch_size=batch_size,
            normalize=normalize,
            timeout=timeout,
        )
