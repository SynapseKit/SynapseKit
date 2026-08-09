"""Shared fixtures for embeddings tests."""

from __future__ import annotations


def embed_payload(model: str, dim: int, texts: list[str]) -> dict:
    """Build an OpenAI-style embedding API response body."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.1] * dim} for i in range(len(texts))
        ],
        "model": model,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }
