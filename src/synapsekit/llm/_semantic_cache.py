"""Semantic LLM Cache: similarity-based cache lookup using embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np


class SemanticCache:
    """Cache LLM responses using semantic similarity instead of exact match.

    Uses embeddings to find semantically similar prompts and returns
    cached responses when similarity exceeds a threshold.

    Vectors are L2-normalised on insertion so lookup reduces to a single
    batched matrix–vector multiply (one BLAS call) instead of a Python
    for-loop over individual dot products.

    Usage::

        from synapsekit.llm._semantic_cache import SemanticCache
        from synapsekit import SynapsekitEmbeddings

        embeddings = SynapsekitEmbeddings()
        cache = SemanticCache(embeddings=embeddings, threshold=0.92)

        # Store a response
        cache.put("What is Python?", "Python is a programming language.")

        # Later, a semantically similar query hits the cache
        result = cache.get("Tell me about Python")
        # result → "Python is a programming language."
    """

    def __init__(
        self,
        embeddings: Any,
        threshold: float = 0.92,
        maxsize: int = 256,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._embeddings = embeddings
        self._threshold = threshold
        self._maxsize = maxsize
        self._entries: list[dict[str, Any]] = []
        # Normalised unit vectors stored individually (easy append/evict)
        self._vectors: list[np.ndarray] = []
        # Stacked matrix rebuilt lazily; None means it needs rebuilding
        self._matrix: np.ndarray | None = None
        self._dirty: bool = False
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0.0 else arr

    async def get(self, prompt: str) -> str | None:
        """Look up a semantically similar prompt in the cache.

        Returns the cached response if similarity >= threshold, else None.
        All cached vectors are L2-normalised, so the cosine similarity matrix
        is computed as a single BLAS matrix–vector multiply instead of a
        Python-level loop.
        """
        if not self._entries:
            self.misses += 1
            return None

        query_vec = await self._embeddings.embed(prompt)
        query_arr = self._normalize(np.array(query_vec, dtype=np.float32))

        # Rebuild the stacked matrix only when entries have changed
        if self._dirty or self._matrix is None:
            self._matrix = np.vstack(self._vectors)  # (n, D) – one allocation
            self._dirty = False

        # One BLAS call replaces the entire Python for-loop
        scores = self._matrix @ query_arr  # (n,) – dot == cosine for unit vecs
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self._threshold:
            self.hits += 1
            result: str = self._entries[best_idx]["response"]
            return result

        self.misses += 1
        return None

    async def put(self, prompt: str, response: str) -> None:
        """Store a prompt-response pair in the cache.

        The embedding is L2-normalised before storage so that cosine
        similarity at lookup time is a plain dot product.
        """
        vec = await self._embeddings.embed(prompt)
        arr = self._normalize(np.array(vec, dtype=np.float32))

        self._entries.append({"prompt": prompt, "response": response})
        self._vectors.append(arr)
        self._dirty = True  # matrix must be rebuilt before next lookup

        # Evict oldest if over maxsize
        if len(self._entries) > self._maxsize:
            self._entries.pop(0)
            self._vectors.pop(0)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
        self._vectors.clear()
        self._matrix = None
        self._dirty = False

    def __len__(self) -> int:
        return len(self._entries)
