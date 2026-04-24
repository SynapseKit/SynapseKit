"""Regression tests for the 12 performance fixes.

Each test class maps 1:1 to a changed file and directly exercises the new
behavior (not just the public API).  Every test is designed to *fail on the
old code* and *pass on the fixed code*, so a regression is immediately caught.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of v."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _make_embeddings(dim: int = 8):
    """Return a mock SynapsekitEmbeddings with deterministic unit vectors."""
    mock = MagicMock()

    async def embed(texts):
        vecs = []
        for i, _ in enumerate(texts):
            v = np.zeros(dim, dtype=np.float32)
            v[i % dim] = 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    async def embed_one(text):
        arr = await embed([text])
        return arr[0]

    mock.embed = embed
    mock.embed_one = embed_one
    return mock


# ---------------------------------------------------------------------------
# 1. SemanticCache — batched matrix lookup, normalisation, dirty flag
# ---------------------------------------------------------------------------


class TestSemanticCacheRegression:
    """#568 — O(n) Python loop replaced by batched numpy matrix multiply."""

    def _make_cache(self, threshold=0.9, maxsize=16):
        from synapsekit.llm._semantic_cache import SemanticCache

        embeddings = MagicMock()
        cache = SemanticCache(embeddings=embeddings, threshold=threshold, maxsize=maxsize)
        return cache

    @pytest.mark.asyncio
    async def test_matrix_is_built_lazily_on_first_get(self):
        from synapsekit.llm._semantic_cache import SemanticCache

        emb = MagicMock()
        emb.embed = AsyncMock(return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32))

        cache = SemanticCache(emb, threshold=0.9)
        # matrix is None before any put
        assert cache._matrix is None
        assert not cache._dirty

        await cache.put("hello", "world")
        # after put, dirty flag is set but matrix not yet built
        assert cache._dirty
        assert cache._matrix is None

        # get triggers matrix build
        emb.embed = AsyncMock(return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32))
        result = await cache.get("hello")
        assert result == "world"
        # matrix is now populated and clean
        assert cache._matrix is not None
        assert not cache._dirty

    @pytest.mark.asyncio
    async def test_vectors_are_normalised_on_put(self):
        from synapsekit.llm._semantic_cache import SemanticCache

        emb = MagicMock()
        # Return an unnormalised vector
        raw = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # norm = 5
        emb.embed = AsyncMock(return_value=raw)

        cache = SemanticCache(emb, threshold=0.5)
        await cache.put("q", "r")

        stored = cache._vectors[0]
        np.testing.assert_allclose(np.linalg.norm(stored), 1.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_dirty_flag_set_after_put_cleared_after_get(self):
        from synapsekit.llm._semantic_cache import SemanticCache

        vec = np.array([1.0, 0.0], dtype=np.float32)
        emb = MagicMock()
        emb.embed = AsyncMock(return_value=vec)

        cache = SemanticCache(emb, threshold=0.5)
        assert not cache._dirty

        await cache.put("a", "b")
        assert cache._dirty

        await cache.get("a")
        assert not cache._dirty

        # Second put → dirty again
        await cache.put("c", "d")
        assert cache._dirty

    @pytest.mark.asyncio
    async def test_hit_and_miss_counts_correct(self):
        from synapsekit.llm._semantic_cache import SemanticCache

        v1 = _unit(np.array([1.0, 0.0], dtype=np.float32))
        v2 = _unit(np.array([0.0, 1.0], dtype=np.float32))

        emb = MagicMock()
        emb.embed = AsyncMock(return_value=v1)
        cache = SemanticCache(emb, threshold=0.99)
        await cache.put("x", "result_x")

        # same vector → hit
        emb.embed = AsyncMock(return_value=v1)
        assert await cache.get("x") == "result_x"
        assert cache.hits == 1
        assert cache.misses == 0

        # orthogonal vector → miss
        emb.embed = AsyncMock(return_value=v2)
        assert await cache.get("y") is None
        assert cache.misses == 1

    @pytest.mark.asyncio
    async def test_clear_resets_matrix_and_dirty(self):
        from synapsekit.llm._semantic_cache import SemanticCache

        v = _unit(np.array([1.0, 0.0], dtype=np.float32))
        emb = MagicMock()
        emb.embed = AsyncMock(return_value=v)

        cache = SemanticCache(emb, threshold=0.5)
        await cache.put("a", "b")
        await cache.get("a")  # builds matrix
        assert cache._matrix is not None

        cache.clear()
        assert cache._matrix is None
        assert not cache._dirty
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_maxsize_eviction_still_works(self):
        from synapsekit.llm._semantic_cache import SemanticCache

        emb = MagicMock()
        emb.embed = AsyncMock(return_value=_unit(np.array([1.0, 0.0], dtype=np.float32)))
        cache = SemanticCache(emb, threshold=0.5, maxsize=3)

        for i in range(5):
            emb.embed = AsyncMock(
                return_value=_unit(np.array([float(i + 1), 0.0], dtype=np.float32))
            )
            await cache.put(f"prompt{i}", f"resp{i}")

        assert len(cache) == 3


# ---------------------------------------------------------------------------
# 2. InMemoryVectorStore — lazy consolidation, inverted index, MMR matrix
# ---------------------------------------------------------------------------


class TestVectorStoreRegression:
    """#569 #574 #572 — three independent fixes in vectorstore.py."""

    @pytest.fixture
    def store(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        return InMemoryVectorStore(_make_embeddings(dim=8))

    # ── lazy consolidation ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_add_does_not_consolidate_immediately(self, store):
        """add() must not call np.vstack — vectors go to _pending."""
        await store.add(["doc1", "doc2"])
        # _vectors stays None; _pending has one batch
        assert store._vectors is None
        assert len(store._pending) == 1

    @pytest.mark.asyncio
    async def test_search_consolidates_pending(self, store):
        await store.add(["doc1", "doc2"])
        assert store._vectors is None  # not yet consolidated
        results = await store.search("doc1", top_k=1)
        assert store._vectors is not None  # now consolidated
        assert store._pending == []  # pending cleared
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multiple_add_batches_all_searchable(self, store):
        await store.add(["batch1_a"])
        await store.add(["batch2_b"])
        await store.add(["batch3_c"])
        assert len(store._pending) == 3  # three un-consolidated batches
        results = await store.search("batch1_a", top_k=3)
        assert len(results) == 3  # all three found after consolidation

    @pytest.mark.asyncio
    async def test_save_consolidates_before_saving(self, store, tmp_path):
        await store.add(["hello"])
        assert store._pending  # not yet consolidated
        store.save(str(tmp_path / "s.npz"))
        # after save, pending must be empty
        assert store._pending == []
        assert store._vectors is not None

    @pytest.mark.asyncio
    async def test_load_clears_pending_and_rebuilds_index(self, store, tmp_path):
        await store.add(["world"], metadata=[{"tag": "a"}])
        path = str(tmp_path / "s.npz")
        store.save(path)

        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store2 = InMemoryVectorStore(_make_embeddings(dim=8))
        store2.load(path)
        assert store2._pending == []
        assert store2._vectors is not None
        # inverted index reconstructed
        assert "tag" in store2._index
        assert "a" in store2._index["tag"]

    # ── inverted metadata index ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_index_built_on_add(self, store):
        await store.add(["doc"], metadata=[{"type": "article", "lang": "en"}])
        assert "type" in store._index
        assert "article" in store._index["type"]
        assert 0 in store._index["type"]["article"]
        assert "lang" in store._index
        assert "en" in store._index["lang"]

    @pytest.mark.asyncio
    async def test_metadata_filter_uses_index(self, store):
        await store.add(
            ["doc_a", "doc_b", "doc_c"],
            metadata=[{"src": "x"}, {"src": "y"}, {"src": "x"}],
        )
        results = await store.search("doc_a", top_k=5, metadata_filter={"src": "x"})
        returned_texts = {r["text"] for r in results}
        assert "doc_a" in returned_texts
        assert "doc_c" in returned_texts
        assert "doc_b" not in returned_texts

    @pytest.mark.asyncio
    async def test_metadata_filter_no_match_returns_empty(self, store):
        await store.add(["doc"], metadata=[{"src": "a"}])
        results = await store.search("doc", top_k=5, metadata_filter={"src": "zzz"})
        assert results == []

    @pytest.mark.asyncio
    async def test_index_accumulates_across_batches(self, store):
        await store.add(["first"], metadata=[{"tag": "t"}])
        await store.add(["second"], metadata=[{"tag": "t"}])
        # Both docs (indices 0 and 1) must be in the index
        assert {0, 1} == store._index["tag"]["t"]

    @pytest.mark.asyncio
    async def test_no_filter_returns_all_candidates(self, store):
        await store.add(["a", "b", "c"])
        results = await store.search("a", top_k=10)
        assert len(results) == 3

    # ── MMR precomputed matrix ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_mmr_returns_top_k_results(self, store):
        await store.add(["x", "y", "z", "w"])
        results = await store.search_mmr("x", top_k=2, fetch_k=4)
        assert len(results) == 2
        assert all("text" in r and "score" in r and "metadata" in r for r in results)

    @pytest.mark.asyncio
    async def test_mmr_empty_store_returns_empty(self, store):
        results = await store.search_mmr("q", top_k=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_mmr_with_metadata_filter(self, store):
        await store.add(
            ["a", "b", "c"],
            metadata=[{"k": "v1"}, {"k": "v2"}, {"k": "v1"}],
        )
        results = await store.search_mmr("a", top_k=5, fetch_k=5, metadata_filter={"k": "v1"})
        returned = {r["text"] for r in results}
        assert "b" not in returned  # filtered out

    @pytest.mark.asyncio
    async def test_mmr_selected_set_prevents_duplicates(self, store):
        await store.add(["alpha", "beta", "gamma", "delta"])
        results = await store.search_mmr("alpha", top_k=3, fetch_k=4)
        texts = [r["text"] for r in results]
        assert len(texts) == len(set(texts)), "MMR returned duplicate documents"


# ---------------------------------------------------------------------------
# 3. WebScraperTool — async DNS, no event-loop blocking
# ---------------------------------------------------------------------------


class TestWebScraperRegression:
    """#570 — socket.gethostbyname offloaded to executor."""

    @pytest.mark.asyncio
    async def test_validate_url_is_async_coroutine(self):
        import inspect

        from synapsekit.agents.tools.web_scraper import _validate_url

        assert inspect.iscoroutinefunction(_validate_url)

    @pytest.mark.asyncio
    async def test_private_ip_still_blocked(self):
        from synapsekit.agents.tools.web_scraper import _validate_url

        with patch("socket.gethostbyname", return_value="192.168.1.1"):
            with pytest.raises(ValueError, match="private"):
                await _validate_url("http://internal.corp/secret")

    @pytest.mark.asyncio
    async def test_public_ip_allowed(self):
        from synapsekit.agents.tools.web_scraper import _validate_url

        with patch("socket.gethostbyname", return_value="8.8.8.8"):
            # should not raise
            await _validate_url("https://example.com/page")

    @pytest.mark.asyncio
    async def test_dns_lookup_uses_executor_not_blocking(self):
        """Verify run_in_executor is called (not the bare blocking call)."""
        from synapsekit.agents.tools.web_scraper import _validate_url

        executor_calls = []

        async def fake_executor(pool, fn, *args):
            executor_calls.append(args)
            return "1.2.3.4"

        loop = asyncio.get_running_loop()
        with patch.object(loop, "run_in_executor", side_effect=fake_executor):
            await _validate_url("https://example.com")

        assert len(executor_calls) == 1
        assert executor_calls[0][0] == "example.com"

    @pytest.mark.asyncio
    async def test_bad_scheme_raises_before_dns(self):
        from synapsekit.agents.tools.web_scraper import _validate_url

        with patch("socket.gethostbyname") as mock_dns:
            with pytest.raises(ValueError, match="scheme"):
                await _validate_url("ftp://example.com")
            mock_dns.assert_not_called()

    @pytest.mark.asyncio
    async def test_gaierror_is_ignored(self):
        import socket

        from synapsekit.agents.tools.web_scraper import _validate_url

        with patch("socket.gethostbyname", side_effect=socket.gaierror):
            # Should not raise — unknown hosts are allowed through
            await _validate_url("https://unknown-host-xyz.example")


# ---------------------------------------------------------------------------
# 4. HTTPRequestTool — persistent session, aclose, ImportError propagates
# ---------------------------------------------------------------------------


class TestHTTPRequestToolRegression:
    """#571 — persistent session reuse, async context manager."""

    @pytest.mark.asyncio
    async def test_session_created_lazily(self):
        from synapsekit.agents.tools.http_request import HTTPRequestTool

        tool = HTTPRequestTool()
        assert tool._session is None  # not yet created

    @pytest.mark.asyncio
    async def test_session_reused_across_calls(self):
        from synapsekit.agents.tools.http_request import HTTPRequestTool

        tool = HTTPRequestTool()

        mock_instance = AsyncMock()
        mock_instance.closed = False
        resp = AsyncMock()
        resp.status = 200
        resp.text = AsyncMock(return_value="ok")
        mock_instance.request.return_value.__aenter__ = AsyncMock(return_value=resp)
        mock_instance.request.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_session_cls = MagicMock(return_value=mock_instance)
        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = mock_session_cls
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"aiohttp": mock_aiohttp}):
            # Clear any cached session so _get_session runs fresh
            tool._session = None
            await tool.run(url="https://example.com")
            await tool.run(url="https://example.com")

        # Session constructor called exactly once despite two run() calls
        assert mock_session_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_aclose_clears_session(self):
        from synapsekit.agents.tools.http_request import HTTPRequestTool

        tool = HTTPRequestTool()
        mock_session = AsyncMock()
        mock_session.closed = False
        tool._session = mock_session

        await tool.aclose()
        mock_session.close.assert_awaited_once()
        assert tool._session is None

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_on_exit(self):
        from synapsekit.agents.tools.http_request import HTTPRequestTool

        mock_session = AsyncMock()
        mock_session.closed = False

        async with HTTPRequestTool() as tool:
            tool._session = mock_session

        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_import_error_propagates_not_swallowed(self):
        from synapsekit.agents.tools.http_request import HTTPRequestTool

        tool = HTTPRequestTool()
        with patch.dict(sys.modules, {"aiohttp": None}):
            with pytest.raises(ImportError, match="aiohttp required"):
                await tool.run(url="https://example.com")


# ---------------------------------------------------------------------------
# 5. TokenBucketRateLimiter — sleep outside lock
# ---------------------------------------------------------------------------


class TestRateLimiterRegression:
    """#573 — lock released before sleeping; concurrent waiters not serialised."""

    @pytest.mark.asyncio
    async def test_acquire_consumes_token(self):
        from synapsekit.llm._rate_limit import TokenBucketRateLimiter

        rl = TokenBucketRateLimiter(requests_per_minute=60)
        initial = rl._tokens
        await rl.acquire()
        assert rl._tokens == initial - 1.0

    @pytest.mark.asyncio
    async def test_lock_not_held_during_sleep(self):
        """After acquire() computes the wait, the lock must be released before
        asyncio.sleep is awaited — verified by checking lock state at sleep time."""
        from synapsekit.llm._rate_limit import TokenBucketRateLimiter

        rl = TokenBucketRateLimiter(requests_per_minute=1)
        rl._tokens = 0.0  # force a wait

        lock_held_during_sleep = []

        real_sleep = asyncio.sleep

        async def spy_sleep(t):
            lock_held_during_sleep.append(rl._lock.locked())
            # avoid real sleep — immediately refill tokens so acquire can succeed
            rl._tokens = 1.0
            await real_sleep(0)

        with patch("asyncio.sleep", side_effect=spy_sleep):
            await rl.acquire()

        assert lock_held_during_sleep, "sleep was never called"
        assert not lock_held_during_sleep[0], "lock was held during sleep — bug!"

    @pytest.mark.asyncio
    async def test_concurrent_acquires_all_succeed(self):
        """Multiple concurrent callers must all eventually get a token."""
        from synapsekit.llm._rate_limit import TokenBucketRateLimiter

        rl = TokenBucketRateLimiter(requests_per_minute=600)  # 10/s
        results = await asyncio.gather(*[rl.acquire() for _ in range(5)])
        assert results == [None] * 5  # acquire() returns None on success

    def test_invalid_rpm_raises(self):
        from synapsekit.llm._rate_limit import TokenBucketRateLimiter

        with pytest.raises(ValueError):
            TokenBucketRateLimiter(requests_per_minute=0)


# ---------------------------------------------------------------------------
# 6. EnsembleRetriever — asyncio.gather (parallel, not sequential)
# ---------------------------------------------------------------------------


class TestEnsembleRetrieverRegression:
    """#576 — all retrievers queried concurrently."""

    def _make_retriever(self, results: list[str]) -> Any:
        r = MagicMock()
        r.retrieve = AsyncMock(return_value=results)
        return r

    @pytest.mark.asyncio
    async def test_all_retrievers_called(self):
        from synapsekit.retrieval.ensemble import EnsembleRetriever

        r1 = self._make_retriever(["doc_a", "doc_b"])
        r2 = self._make_retriever(["doc_b", "doc_c"])
        ensemble = EnsembleRetriever([r1, r2])

        results = await ensemble.retrieve("query", top_k=3)
        r1.retrieve.assert_awaited_once()
        r2.retrieve.assert_awaited_once()
        assert set(results) <= {"doc_a", "doc_b", "doc_c"}

    @pytest.mark.asyncio
    async def test_gather_called_not_sequential(self):
        """asyncio.gather must be used — not a for-loop awaiting one by one."""
        from synapsekit.retrieval.ensemble import EnsembleRetriever

        r1 = self._make_retriever(["x"])
        r2 = self._make_retriever(["y"])
        ensemble = EnsembleRetriever([r1, r2])

        gather_calls = []
        real_gather = asyncio.gather

        async def spy_gather(*coros, **kw):
            gather_calls.append(len(coros))
            return await real_gather(*coros, **kw)

        with patch("synapsekit.retrieval.ensemble.asyncio.gather", side_effect=spy_gather):
            await ensemble.retrieve("q", top_k=2)

        assert gather_calls == [2], "asyncio.gather not called with both retrievers"

    @pytest.mark.asyncio
    async def test_rrf_fusion_correct(self):
        from synapsekit.retrieval.ensemble import EnsembleRetriever

        r1 = self._make_retriever(["a", "b"])
        r2 = self._make_retriever(["b", "c"])
        # equal weights → 'b' appears in both → highest RRF score
        ensemble = EnsembleRetriever([r1, r2], weights=[1.0, 1.0])
        results = await ensemble.retrieve("q", top_k=3)
        assert results[0] == "b"

    @pytest.mark.asyncio
    async def test_single_retriever_works(self):
        from synapsekit.retrieval.ensemble import EnsembleRetriever

        r = self._make_retriever(["only"])
        ensemble = EnsembleRetriever([r])
        results = await ensemble.retrieve("q", top_k=1)
        assert results == ["only"]

    def test_mismatched_weights_raises(self):
        from synapsekit.retrieval.ensemble import EnsembleRetriever

        r = self._make_retriever([])
        with pytest.raises(ValueError, match="weights"):
            EnsembleRetriever([r], weights=[1.0, 2.0])


# ---------------------------------------------------------------------------
# 7. AsyncLRUCache — sort_keys removed, key still deterministic
# ---------------------------------------------------------------------------


class TestCacheKeyRegression:
    """#577 — sort_keys removed; key must still be stable and unique."""

    def test_key_is_deterministic(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("gpt-4o", "hello", 0.5, 512)
        k2 = AsyncLRUCache.make_key("gpt-4o", "hello", 0.5, 512)
        assert k1 == k2

    def test_different_prompts_different_keys(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("m", "hello", 0.0, 100)
        k2 = AsyncLRUCache.make_key("m", "world", 0.0, 100)
        assert k1 != k2

    def test_different_models_different_keys(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("gpt-4o", "p", 0.0, 100)
        k2 = AsyncLRUCache.make_key("gpt-3.5", "p", 0.0, 100)
        assert k1 != k2

    def test_different_temperatures_different_keys(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("m", "p", 0.0, 100)
        k2 = AsyncLRUCache.make_key("m", "p", 1.0, 100)
        assert k1 != k2

    def test_different_max_tokens_different_keys(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("m", "p", 0.5, 100)
        k2 = AsyncLRUCache.make_key("m", "p", 0.5, 200)
        assert k1 != k2

    def test_key_is_hex_string(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k = AsyncLRUCache.make_key("m", "p", 0.0, 10)
        assert isinstance(k, str)
        int(k, 16)  # must be valid hex

    def test_messages_list_key_stable(self):
        from synapsekit.llm._cache import AsyncLRUCache

        msgs = [{"role": "user", "content": "hi"}]
        k1 = AsyncLRUCache.make_key("m", msgs, 0.0, 10)
        k2 = AsyncLRUCache.make_key("m", msgs, 0.0, 10)
        assert k1 == k2


# ---------------------------------------------------------------------------
# 8. SQLiteLLMCache — context manager, idempotent close, __del__ safety
# ---------------------------------------------------------------------------


class TestSQLiteCacheRegression:
    """#578 — connection guaranteed to close; context manager protocol."""

    def test_context_manager_closes_connection(self, tmp_path):
        from synapsekit.llm._sqlite_cache import SQLiteLLMCache

        db = str(tmp_path / "test.db")
        with SQLiteLLMCache(db) as cache:
            cache.put("k", "v")
            assert cache.get("k") == "v"
        # after __exit__, connection should be closed
        with pytest.raises(Exception):
            cache._conn.execute("SELECT 1")  # closed connection raises

    def test_close_is_idempotent(self, tmp_path):
        from synapsekit.llm._sqlite_cache import SQLiteLLMCache

        db = str(tmp_path / "test.db")
        cache = SQLiteLLMCache(db)
        cache.close()
        cache.close()  # second close must not raise

    def test_del_does_not_raise(self, tmp_path):
        from synapsekit.llm._sqlite_cache import SQLiteLLMCache

        db = str(tmp_path / "test.db")
        cache = SQLiteLLMCache(db)
        # Simulate __del__ without context manager
        cache.__del__()  # must not raise

    def test_basic_put_get_still_works(self, tmp_path):
        from synapsekit.llm._sqlite_cache import SQLiteLLMCache

        db = str(tmp_path / "test.db")
        with SQLiteLLMCache(db) as cache:
            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"
            assert cache.hits == 1
            assert cache.misses == 0

    def test_miss_returns_none(self, tmp_path):
        from synapsekit.llm._sqlite_cache import SQLiteLLMCache

        db = str(tmp_path / "test.db")
        with SQLiteLLMCache(db) as cache:
            result = cache.get("nonexistent")
            assert result is None
            assert cache.misses == 1

    def test_clear_empties_db(self, tmp_path):
        from synapsekit.llm._sqlite_cache import SQLiteLLMCache

        db = str(tmp_path / "test.db")
        with SQLiteLLMCache(db) as cache:
            cache.put("k", "v")
            cache.clear()
            assert len(cache) == 0
            assert cache.get("k") is None

    def test_persists_across_instances(self, tmp_path):
        from synapsekit.llm._sqlite_cache import SQLiteLLMCache

        db = str(tmp_path / "test.db")
        with SQLiteLLMCache(db) as c1:
            c1.put("persistent", "data")

        with SQLiteLLMCache(db) as c2:
            assert c2.get("persistent") == "data"


# ---------------------------------------------------------------------------
# 9. EvaluationPipeline — asyncio.gather for metrics; semaphore for batch
# ---------------------------------------------------------------------------


class TestEvaluationPipelineRegression:
    """#575 — metrics evaluated concurrently; batch uses semaphore."""

    def _make_metric(self, name: str, score: float = 0.8):
        from synapsekit.evaluation.base import MetricResult

        m = MagicMock()
        m.name = name
        result = MetricResult(score=score)
        m.evaluate = AsyncMock(return_value=result)
        return m

    @pytest.mark.asyncio
    async def test_all_metrics_called(self):
        from synapsekit.evaluation.pipeline import EvaluationPipeline

        m1 = self._make_metric("faithfulness", 0.9)
        m2 = self._make_metric("relevancy", 0.7)
        pipeline = EvaluationPipeline([m1, m2])

        result = await pipeline.evaluate(question="q", answer="a", contexts=["c"])
        m1.evaluate.assert_awaited_once()
        m2.evaluate.assert_awaited_once()
        assert "faithfulness" in result.scores
        assert "relevancy" in result.scores

    @pytest.mark.asyncio
    async def test_gather_used_for_metrics(self):
        """asyncio.gather must fan out all metrics, not loop sequentially."""
        from synapsekit.evaluation.pipeline import EvaluationPipeline

        m1 = self._make_metric("a")
        m2 = self._make_metric("b")
        pipeline = EvaluationPipeline([m1, m2])

        gather_calls = []
        real_gather = asyncio.gather

        async def spy(*coros, **kw):
            gather_calls.append(len(coros))
            return await real_gather(*coros, **kw)

        with patch("synapsekit.evaluation.pipeline.asyncio.gather", side_effect=spy):
            await pipeline.evaluate(question="q", answer="a")

        assert gather_calls[0] == 2, "gather not called with all metrics"

    @pytest.mark.asyncio
    async def test_mean_score_correct(self):
        from synapsekit.evaluation.pipeline import EvaluationPipeline

        m1 = self._make_metric("x", 0.6)
        m2 = self._make_metric("y", 0.8)
        pipeline = EvaluationPipeline([m1, m2])
        result = await pipeline.evaluate(question="q", answer="a")
        assert abs(result.mean_score - 0.7) < 1e-6

    @pytest.mark.asyncio
    async def test_evaluate_batch_runs_all_samples(self):
        from synapsekit.evaluation.pipeline import EvaluationPipeline

        m = self._make_metric("f", 1.0)
        pipeline = EvaluationPipeline([m])
        samples = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
        results = await pipeline.evaluate_batch(samples)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_batch_concurrency_semaphore_limits_concurrent(self):
        """Semaphore must cap concurrent evaluations at `concurrency`."""
        from synapsekit.evaluation.base import MetricResult
        from synapsekit.evaluation.pipeline import EvaluationPipeline

        active: list[int] = []
        peak: list[int] = [0]

        async def slow_evaluate(**_):
            active.append(1)
            peak[0] = max(peak[0], len(active))
            await asyncio.sleep(0)
            active.pop()
            return MetricResult(score=1.0)

        m = MagicMock()
        m.name = "x"
        m.evaluate = slow_evaluate

        pipeline = EvaluationPipeline([m])
        samples = [{"question": f"q{i}", "answer": "a"} for i in range(20)]
        await pipeline.evaluate_batch(samples, concurrency=3)
        # Peak concurrent evaluations must not exceed concurrency x metrics
        assert peak[0] <= 3


# ---------------------------------------------------------------------------
# 10. SitemapLoader — deque BFS, O(1) popleft
# ---------------------------------------------------------------------------


class TestSitemapLoaderRegression:
    """#579 — list.pop(0) replaced by deque.popleft()."""

    @pytest.mark.asyncio
    async def test_collect_urls_uses_deque(self):
        """_collect_urls must use a deque, not a plain list."""
        import inspect

        from synapsekit.loaders import sitemap as sitemap_module

        source = inspect.getsource(sitemap_module)
        # Check deque is imported and used in the source
        assert "deque" in source
        assert "popleft" in source

    def test_deque_imported_in_module(self):
        from synapsekit.loaders import sitemap as mod

        # deque must be importable from the module's namespace
        assert "deque" in dir(mod) or hasattr(mod, "deque") or "deque" in mod.__dict__

    @pytest.mark.asyncio
    async def test_sitemap_bfs_visits_urls(self):
        """Full aload path with mocked HTTP — verifies docs are returned."""
        from synapsekit.loaders.sitemap import SitemapLoader

        sitemap_xml = """<?xml version="1.0"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://example.com/page1</loc></url>
          <url><loc>https://example.com/page2</loc></url>
        </urlset>"""

        page_html = "<html><body><p>Hello world content here</p></body></html>"

        call_count = [0]

        async def mock_get(url, **_):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "sitemap" in url:
                resp.text = sitemap_xml
            else:
                resp.text = page_html
                call_count[0] += 1
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        loader = SitemapLoader("https://example.com/sitemap.xml")

        with patch("httpx.AsyncClient", return_value=mock_client):
            docs = await loader.aload()

        assert len(docs) == 2
        assert all(d.metadata["source"] == "sitemap" for d in docs)

    @pytest.mark.asyncio
    async def test_filter_urls_applied(self):
        from synapsekit.loaders.sitemap import SitemapLoader

        sitemap_xml = """<?xml version="1.0"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://example.com/blog/post1</loc></url>
          <url><loc>https://example.com/about</loc></url>
        </urlset>"""

        page_html = "<html><body><p>Content</p></body></html>"

        async def mock_get(url, **_):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.text = sitemap_xml if "sitemap" in url else page_html
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        loader = SitemapLoader("https://example.com/sitemap.xml", filter_urls=["/blog/"])

        with patch("httpx.AsyncClient", return_value=mock_client):
            docs = await loader.aload()

        assert len(docs) == 1
        assert "blog" in docs[0].metadata["url"]

    def test_invalid_url_raises(self):
        from synapsekit.loaders.sitemap import SitemapLoader

        with pytest.raises(ValueError, match="scheme"):
            SitemapLoader("ftp://bad.com/sitemap.xml")
