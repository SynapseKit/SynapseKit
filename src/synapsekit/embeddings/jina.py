"""Jina AI embeddings provider (jina-embeddings-v3, long-context, task LoRA)."""

from __future__ import annotations

from .http import HTTPEmbeddings


class JinaEmbeddings(HTTPEmbeddings):
    """Async embeddings backed by the Jina AI Embeddings API.

    ``jina-embeddings-v3`` supports long contexts (8k tokens) and task LoRA
    via the ``task`` field — e.g. ``"retrieval.passage"`` for ingestion and
    ``"retrieval.query"`` for queries.

    Usage::

        emb = JinaEmbeddings(api_key="jina_...")            # or JINA_API_KEY
        emb = JinaEmbeddings(task="retrieval.passage", dimensions=1024)
        vecs = await emb.embed(["hello", "world"])

    Requires ``httpx``: ``pip install synapsekit[jina]``
    """

    dimensions: int | None = 1024

    def __init__(
        self,
        model: str = "jina-embeddings-v3",
        *,
        api_key: str | None = None,
        task: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        timeout: float = 60.0,
    ) -> None:
        request_extra: dict[str, str | int] = {}
        if task is not None:
            request_extra["task"] = task
        if dimensions is not None:
            request_extra["dimensions"] = dimensions
            self.dimensions = dimensions
        super().__init__(
            model,
            api_key=api_key,
            base_url="https://api.jina.ai/v1",
            env_key="JINA_API_KEY",
            batch_size=batch_size,
            normalize=normalize,
            timeout=timeout,
            **request_extra,
        )
