from __future__ import annotations

from typing import Any

import numpy as np

from .._json import dumps as _json_dumps
from .._json import loads as _json_loads
from ..embeddings.backend import SynapsekitEmbeddings
from .base import VectorStore

try:
    from .._rust_core import deserialize_metadata_list as _rust_deser
    from .._rust_core import serialize_metadata_list as _rust_ser
except ImportError:
    _rust_ser = None
    _rust_deser = None

# Growth factor for pre-allocated vector buffer
_GROWTH_FACTOR = 2


class InMemoryVectorStore(VectorStore):
    """
    Numpy-backed in-memory vector store.
    Supports cosine similarity search, save/load via .npz.

    Performance design
    ------------------
    * **O(1) amortised inserts** — newly embedded batches are queued in
      ``_pending`` and merged into a pre-allocated buffer (doubling strategy)
      only when a search is issued, avoiding per-search ``np.vstack`` copies.

    * **O(result) metadata filtering** — an inverted index maps each
      ``field → value → set[doc_idx]`` so that filter queries intersect small
      sets instead of scanning all documents.

    * **Vectorised MMR** — the greedy selection loop uses numpy masked arrays
      to compute MMR scores for all candidates at once, replacing the inner
      Python ``for`` loop with a single vectorised pass per selection round.
    """

    def __init__(self, embedding_backend: SynapsekitEmbeddings) -> None:
        self._embeddings = embedding_backend
        # Pre-allocated buffer with doubling strategy
        self._buf: np.ndarray | None = None  # (capacity, D)
        self._consolidated: int = 0  # rows in _buf that are consolidated
        self._vectors: np.ndarray | None = None  # view of _buf[:_consolidated]
        # Newly added batches waiting to be merged
        self._pending: list[np.ndarray] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []
        # Inverted index: field → value → set of global doc indices
        self._index: dict[str, dict[Any, set[int]]] = {}

    # ── internal helpers ────────────────────────────────────────────────────

    def _ensure_buf_capacity(self, total_needed: int, dim: int) -> None:
        """Ensure _buf can hold at least ``total_needed`` rows."""
        if self._buf is not None and self._buf.shape[0] >= total_needed:
            return
        new_cap = max(256, total_needed)
        if self._buf is not None:
            cap = self._buf.shape[0]
            while cap < total_needed:
                cap *= _GROWTH_FACTOR
            new_cap = cap
            new_buf = np.empty((new_cap, dim), dtype=np.float32)
            new_buf[: self._consolidated] = self._buf[: self._consolidated]
        else:
            new_buf = np.empty((new_cap, dim), dtype=np.float32)
        self._buf = new_buf

    def _consolidate(self) -> None:
        """Merge pending batches into the pre-allocated buffer.

        Called lazily before any search so that ``add()`` itself is O(chunk).
        Uses a doubling buffer to avoid O(n) copies on every consolidation.
        """
        if not self._pending:
            return
        total_new = sum(p.shape[0] for p in self._pending)
        dim = self._pending[0].shape[1]
        self._ensure_buf_capacity(self._consolidated + total_new, dim)
        assert self._buf is not None
        for batch in self._pending:
            n = batch.shape[0]
            self._buf[self._consolidated : self._consolidated + n] = batch
            self._consolidated += n
        self._pending.clear()
        self._vectors = self._buf[: self._consolidated]

    def _update_index(self, meta: list[dict], base: int) -> None:
        """Add ``len(meta)`` new documents to the inverted index."""
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
        candidate_sets = [self._index.get(k, {}).get(v, set()) for k, v in metadata_filter.items()]
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
        """Returns top_k results sorted by cosine similarity (desc)."""
        from ..observe.runtime import end_span, record_exception, start_span

        search_span = start_span(
            "vector_store.search",
            {
                "vector_store.type": type(self).__name__,
                "vector_store.top_k": top_k,
            },
        )
        try:
            if not self._texts:
                end_span(search_span, attributes={"vector_store.results": 0})
                return []

            self._consolidate()
            assert self._vectors is not None  # guaranteed after consolidate + non-empty texts

            q_vec = await self._embeddings.embed_one(query)  # (D,)
            scores = self._vectors @ q_vec  # (N,) cosine sim (vecs are L2-normalised)

            candidates = self._filter_candidates(metadata_filter or {})

            if candidates is not None:
                # metadata_filter was active
                if not candidates:
                    end_span(search_span, attributes={"vector_store.results": 0})
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

            payload = [
                {
                    "text": self._texts[i],
                    "score": float(scores[i]),
                    "metadata": self._metadata[i],
                }
                for i in top_indices
            ]
            end_span(search_span, attributes={"vector_store.results": len(payload)})
            return payload
        except Exception as exc:
            record_exception(search_span, exc)
            end_span(search_span, error=exc)
            raise

    async def search_mmr(
        self,
        query: str,
        top_k: int = 5,
        lambda_mult: float = 0.5,
        fetch_k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Maximal Marginal Relevance search (vectorised greedy loop).

        Greedily selects documents that maximize:
        ``lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))``

        The inner candidate scoring is fully vectorised with numpy —
        no Python-level inner loop over candidates.
        """
        from ..observe.runtime import end_span, record_exception, start_span

        search_span = start_span(
            "vector_store.search",
            {
                "vector_store.type": f"{type(self).__name__}.mmr",
                "vector_store.top_k": top_k,
            },
        )
        try:
            if not self._texts:
                end_span(search_span, attributes={"vector_store.results": 0})
                return []

            self._consolidate()
            assert self._vectors is not None

            q_vec = await self._embeddings.embed_one(query)  # (D,)
            scores = self._vectors @ q_vec  # (N,)

            candidates = self._filter_candidates(metadata_filter or {})
            if candidates is not None:
                if not candidates:
                    end_span(search_span, attributes={"vector_store.results": 0})
                    return []
            else:
                candidates = list(range(len(self._texts)))

            candidate_scores = sorted(
                ((i, float(scores[i])) for i in candidates),
                key=lambda x: x[1],
                reverse=True,
            )
            pool = candidate_scores[: min(fetch_k, len(candidate_scores))]

            if not pool:
                end_span(search_span, attributes={"vector_store.results": 0})
                return []

            pool_indices = [idx for idx, _ in pool]
            pool_vecs = self._vectors[pool_indices]
            sim_matrix = pool_vecs @ pool_vecs.T  # (P, P)
            rel_scores = np.array([s for _, s in pool], dtype=np.float64)  # (P,)

            n_pool = len(pool)
            n_select = min(top_k, n_pool)
            alive = np.ones(n_pool, dtype=bool)
            max_sim_to_selected = np.full(n_pool, 0.0, dtype=np.float64)

            selected: list[int] = []

            for _ in range(n_select):
                # Vectorised MMR score for all alive candidates
                mmr = lambda_mult * rel_scores - (1 - lambda_mult) * max_sim_to_selected
                mmr[~alive] = -np.inf
                best_pos = int(np.argmax(mmr))
                if mmr[best_pos] == -np.inf:
                    break

                selected.append(pool_indices[best_pos])
                alive[best_pos] = False

                # Update max similarity to selected set
                np.maximum(max_sim_to_selected, sim_matrix[best_pos], out=max_sim_to_selected)

            payload = [
                {
                    "text": self._texts[i],
                    "score": float(scores[i]),
                    "metadata": self._metadata[i],
                }
                for i in selected
            ]
            end_span(search_span, attributes={"vector_store.results": len(payload)})
            return payload
        except Exception as exc:
            record_exception(search_span, exc)
            end_span(search_span, error=exc)
            raise

    def save(self, path: str) -> None:
        """Persist vectors, texts, and metadata to a .npz file."""
        self._consolidate()
        if self._vectors is None:
            raise ValueError("Nothing to save — store is empty.")

        np.savez(
            path,
            vectors=self._vectors,
            texts=np.array(self._texts, dtype=object),
            metadata=np.array(
                _rust_ser(self._metadata)
                if _rust_ser is not None
                else [_json_dumps(m) for m in self._metadata],
                dtype=object,
            ),
        )

    def load(self, path: str) -> None:
        """Load vectors, texts, and metadata from a .npz file."""
        data = np.load(path, allow_pickle=True)
        loaded_vecs = data["vectors"].astype(np.float32)
        n, dim = loaded_vecs.shape
        cap = max(256, n)
        self._buf = np.empty((cap, dim), dtype=np.float32)
        self._buf[:n] = loaded_vecs
        self._consolidated = n
        self._vectors = self._buf[:n]
        self._texts = list(data["texts"])
        raw_meta = list(data["metadata"])
        self._metadata = (
            _rust_deser(raw_meta) if _rust_deser is not None
            else [_json_loads(s) for s in raw_meta]
        )
        self._pending.clear()

        # Rebuild inverted index from loaded metadata
        self._index.clear()
        self._update_index(self._metadata, base=0)

    def __len__(self) -> int:
        return len(self._texts)
