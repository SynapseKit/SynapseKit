from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore


class InMemoryVectorStore(VectorStore):
    """
    Numpy-backed in-memory vector store.
    Supports cosine similarity search, save/load via .npz.

    Performance design
    ------------------
    * **O(1) amortised inserts** — newly embedded batches are queued in
      ``_pending`` and merged into the consolidated matrix only when a search
      is issued (lazy consolidation).  The old ``np.concatenate`` on every
      ``add()`` caused O(n²) total copies for n documents.

    * **O(result) metadata filtering** — an inverted index maps each
      ``field → value → set[doc_idx]`` so that filter queries intersect small
      sets instead of scanning all documents.

    * **O(fetch_k²) MMR precomputation** — the pairwise similarity matrix for
      the candidate pool is computed once with a single BLAS call before the
      greedy selection loop, replacing O(top_k × fetch_k × selected) Python-
      level dot products.
    """

    def __init__(self, embedding_backend: SynapsekitEmbeddings) -> None:
        self._embeddings = embedding_backend
        # Consolidated matrix — updated lazily before every search
        self._vectors: np.ndarray | None = None  # (N, D)
        # Newly added batches waiting to be merged
        self._pending: list[np.ndarray] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        # Inverted index: field → value → set of global doc indices
        self._index: dict[str, dict[Any, set[int]]] = {}

    # ── internal helpers ────────────────────────────────────────────────────

    def _consolidate(self) -> None:
        """Merge pending batches into the consolidated matrix.

        Called lazily before any search so that ``add()`` itself is O(chunk).
        """
        if not self._pending:
            return
        if self._vectors is None:
            self._vectors = np.vstack(self._pending)
        else:
            self._vectors = np.vstack([self._vectors, *self._pending])
        self._pending.clear()

    def _update_index(self, meta: list[dict], base: int) -> None:
        """Add ``len(meta)`` new documents to the inverted index.

        Args:
            meta: Metadata dicts for the new documents.
            base: Global index of the first new document (``len(self._texts)``
                  before the new texts were appended).
        """
        for j, m in enumerate(meta):
            global_idx = base + j
            for k, v in m.items():
                if k not in self._index:
                    self._index[k] = {}
                bucket = self._index[k]
                if v not in bucket:
                    bucket[v] = set()
                bucket[v].add(global_idx)

    def _filter_candidates(self, metadata_filter: dict) -> list[int] | None:
        """Return sorted candidate indices matching *all* filter key-value pairs.

        Returns ``None`` if no filter is active (caller uses all documents).
        Returns an empty list if the filter matches nothing.
        Uses set intersection — O(result_size) not O(n).
        """
        if not metadata_filter:
            return None
        candidate_sets = [
            self._index.get(k, {}).get(v, set()) for k, v in metadata_filter.items()
        ]
        if not candidate_sets or any(not s for s in candidate_sets):
            return []
        return sorted(set.intersection(*candidate_sets))

    # ── public API ──────────────────────────────────────────────────────────

    async def add(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        if not texts:
            return
        meta = metadata or [{} for _ in texts]
        vecs = await self._embeddings.embed(texts)  # (len(texts), D)

        base = len(self._texts)
        self._texts.extend(texts)
        self._metadata.extend(meta)
        # Queue for lazy consolidation — O(1) instead of O(n) copy
        self._pending.append(vecs)
        # Update inverted index immediately so filters work before consolidation
        self._update_index(meta, base)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Returns top_k results sorted by cosine similarity (desc).

        Args:
            metadata_filter: If provided, only include documents whose metadata
                contains all the specified key-value pairs.
        """
        if not self._texts:
            return []

        self._consolidate()
        assert self._vectors is not None  # guaranteed after consolidate + non-empty texts

        q_vec = await self._embeddings.embed_one(query)  # (D,)
        scores = self._vectors @ q_vec  # (N,) cosine sim (vecs are L2-normalised)

        candidates = self._filter_candidates(metadata_filter or {})

        if candidates is not None:
            # metadata_filter was active
            if not candidates:
                return []
            candidate_arr = np.array(candidates, dtype=np.intp)
            candidate_scores = scores[candidate_arr]
            k = min(top_k, len(candidates))
            local_top = np.argpartition(candidate_scores, -k)[-k:]
            local_top = local_top[np.argsort(candidate_scores[local_top])[::-1]]
            top_indices = [candidates[j] for j in local_top]
        else:
            k = min(top_k, len(self._texts))
            _top = np.argpartition(scores, -k)[-k:]
            top_indices = _top[np.argsort(scores[_top])[::-1]].tolist()

        return [
            {
                "text": self._texts[i],
                "score": float(scores[i]),
                "metadata": self._metadata[i],
            }
            for i in top_indices
        ]

    async def search_mmr(
        self,
        query: str,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Maximal Marginal Relevance search.

        Greedily selects documents that maximize:
        ``lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))``

        The pairwise similarity matrix for the candidate pool is precomputed
        with a single BLAS call before the greedy loop, replacing the previous
        O(top_k × fetch_k × selected) Python-level dot-product recomputation.
        """
        if not self._texts:
            return []

        self._consolidate()
        assert self._vectors is not None

        q_vec = await self._embeddings.embed_one(query)  # (D,)
        scores = self._vectors @ q_vec  # (N,)

        # Build candidate pool using the inverted index when a filter is set
        candidates = self._filter_candidates(metadata_filter or {})
        if candidates is not None:
            if not candidates:
                return []
        else:
            candidates = list(range(len(self._texts)))

        # Take top fetch_k by relevance
        candidate_scores = sorted(
            ((i, float(scores[i])) for i in candidates),
            key=lambda x: x[1],
            reverse=True,
        )
        pool = candidate_scores[: min(fetch_k, len(candidate_scores))]

        if not pool:
            return []

        pool_indices = [idx for idx, _ in pool]
        pool_rel_scores = [rel for _, rel in pool]

        # ── precompute pairwise similarity matrix for the pool (one BLAS call) ──
        pool_vecs = self._vectors[pool_indices]     # (fetch_k, D)
        sim_matrix = pool_vecs @ pool_vecs.T         # (fetch_k, fetch_k)

        # Greedy MMR selection
        selected: list[int] = []           # global indices of selected docs
        selected_pos: list[int] = []       # corresponding positions in pool

        selected_set: set[int] = set()     # O(1) membership test

        for _ in range(min(top_k, len(pool))):
            best_global_idx = -1
            best_score = float("-inf")
            best_pool_pos = -1

            for pos, (global_idx, rel_score) in enumerate(pool):
                if global_idx in selected_set:
                    continue

                if selected_pos:
                    # Max sim to already-selected docs — single numpy op on a row slice
                    sim_to_selected = float(np.max(sim_matrix[pos, selected_pos]))
                else:
                    sim_to_selected = 0.0

                mmr_score = lambda_mult * rel_score - (1 - lambda_mult) * sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_global_idx = global_idx
                    best_pool_pos = pos

            if best_global_idx == -1:
                break
            selected.append(best_global_idx)
            selected_pos.append(best_pool_pos)
            selected_set.add(best_global_idx)

        return [
            {
                "text": self._texts[i],
                "score": float(scores[i]),
                "metadata": self._metadata[i],
            }
            for i in selected
        ]

    def save(self, path: str) -> None:
        """Persist vectors, texts, and metadata to a .npz file."""
        self._consolidate()
        if self._vectors is None:
            raise ValueError("Nothing to save — store is empty.")

        np.savez(
            path,
            vectors=self._vectors,
            texts=np.array(self._texts, dtype=object),
            metadata=np.array([json.dumps(m) for m in self._metadata], dtype=object),
        )

    def load(self, path: str) -> None:
        """Load vectors, texts, and metadata from a .npz file."""
        data = np.load(path, allow_pickle=True)
        self._vectors = data["vectors"].astype(np.float32)
        self._texts = list(data["texts"])
        self._metadata = [json.loads(s) for s in data["metadata"]]
        self._pending.clear()

        # Rebuild inverted index from loaded metadata
        self._index.clear()
        self._update_index(self._metadata, base=0)

    def __len__(self) -> int:
        return len(self._texts)
