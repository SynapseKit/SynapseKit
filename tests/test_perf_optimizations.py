"""Production-level tests for performance optimizations.

Covers: _json module, _loop module, xxhash cache keys, __slots__,
pre-allocated vector buffer, vectorised MMR, and checkpointer JSON wiring.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# 1. _json module — orjson/stdlib fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestJsonDumps:
    """Branch coverage for synapsekit._json.dumps."""

    def test_returns_str(self):
        from synapsekit._json import dumps

        result = dumps({"a": 1})
        assert isinstance(result, str)

    def test_roundtrip_dict(self):
        from synapsekit._json import dumps, loads

        obj = {"key": "value", "num": 42, "flag": True, "nil": None}
        assert loads(dumps(obj)) == obj

    def test_roundtrip_list(self):
        from synapsekit._json import dumps, loads

        obj = [1, "two", 3.0, None, False]
        assert loads(dumps(obj)) == obj

    def test_nested_deeply(self):
        from synapsekit._json import dumps, loads

        obj = {"a": {"b": {"c": {"d": [1, 2, {"e": "f"}]}}}}
        assert loads(dumps(obj)) == obj

    def test_empty_dict(self):
        from synapsekit._json import dumps, loads

        assert loads(dumps({})) == {}

    def test_empty_list(self):
        from synapsekit._json import dumps, loads

        assert loads(dumps([])) == []

    def test_unicode_roundtrip(self):
        from synapsekit._json import dumps, loads

        obj = {"emoji": "\U0001f600\U0001f680", "cjk": "\u4e16\u754c", "arabic": "\u0627\u0644\u0639\u0631\u0628\u064a\u0629"}
        assert loads(dumps(obj)) == obj

    def test_special_float_values(self):
        """inf/nan: orjson raises, stdlib json produces non-standard output."""
        from synapsekit._json import dumps

        # Either raises or produces a string (stdlib json allows inf by default)
        try:
            result = dumps(float("inf"))
            assert isinstance(result, str)  # stdlib produces "Infinity"
        except (ValueError, OverflowError):
            pass  # orjson raises — also acceptable

    def test_large_integer(self):
        from synapsekit._json import dumps, loads

        obj = {"big": 2**53}
        assert loads(dumps(obj)) == obj

    def test_string_with_escapes(self):
        from synapsekit._json import dumps, loads

        obj = {"text": 'line1\nline2\ttab "quoted" \\backslash'}
        assert loads(dumps(obj)) == obj


class TestJsonDumpsBytes:
    def test_returns_bytes(self):
        from synapsekit._json import dumps_bytes

        result = dumps_bytes({"x": 1})
        assert isinstance(result, bytes)

    def test_roundtrip_through_loads(self):
        from synapsekit._json import dumps_bytes, loads

        obj = {"key": [1, 2, 3]}
        assert loads(dumps_bytes(obj)) == obj

    def test_decodable_as_utf8(self):
        from synapsekit._json import dumps_bytes

        result = dumps_bytes({"emoji": "\U0001f600"})
        decoded = result.decode("utf-8")
        assert "\U0001f600" in decoded or "\\u" in decoded  # either literal or escaped


class TestJsonLoads:
    def test_from_str(self):
        from synapsekit._json import loads

        assert loads('{"a": 1}') == {"a": 1}

    def test_from_bytes(self):
        from synapsekit._json import loads

        assert loads(b'{"a": 1}') == {"a": 1}

    def test_array_input(self):
        from synapsekit._json import loads

        assert loads("[1,2,3]") == [1, 2, 3]

    def test_scalar_string(self):
        from synapsekit._json import loads

        assert loads('"hello"') == "hello"

    def test_scalar_number(self):
        from synapsekit._json import loads

        assert loads("42") == 42

    def test_invalid_json_raises(self):
        from synapsekit._json import loads

        with pytest.raises((ValueError, TypeError)):
            loads("{invalid")

    def test_empty_string_raises(self):
        from synapsekit._json import loads

        with pytest.raises((ValueError, TypeError)):
            loads("")

    def test_none_raises(self):
        from synapsekit._json import loads

        with pytest.raises((ValueError, TypeError)):
            loads(None)  # type: ignore[arg-type]


class TestJsonStress:
    def test_large_dict_roundtrip(self):
        """1000-key dict must survive roundtrip."""
        from synapsekit._json import dumps, loads

        obj = {f"key_{i}": i * 0.1 for i in range(1000)}
        result = loads(dumps(obj))
        assert len(result) == 1000
        assert isinstance(result["key_500"], float)

    def test_large_nested_list(self):
        """Deeply nested list of 10k items."""
        from synapsekit._json import dumps, loads

        obj = [[i, str(i)] for i in range(10000)]
        result = loads(dumps(obj))
        assert len(result) == 10000
        assert result[9999] == [9999, "9999"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. _loop module — uvloop installer
# ═══════════════════════════════════════════════════════════════════════════


class TestInstallFastLoop:
    def test_no_error(self):
        from synapsekit._loop import install_fast_loop

        install_fast_loop()

    def test_idempotent(self):
        from synapsekit._loop import install_fast_loop

        install_fast_loop()
        install_fast_loop()
        install_fast_loop()

    def test_returns_none(self):
        from synapsekit._loop import install_fast_loop

        assert install_fast_loop() is None


# ═══════════════════════════════════════════════════════════════════════════
# 3. Cache key hashing — xxhash/sha256 fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestCacheKeyHashing:
    def test_deterministic(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        k2 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        assert k1 == k2

    def test_returns_str(self):
        from synapsekit.llm._cache import AsyncLRUCache

        key = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        assert isinstance(key, str)

    def test_hex_format(self):
        from synapsekit.llm._cache import AsyncLRUCache

        key = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        assert len(key) in (32, 64)
        int(key, 16)  # Must be valid hex

    def test_different_model_different_key(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        k2 = AsyncLRUCache.make_key("gpt-3.5", "hello", 0.7, 100)
        assert k1 != k2

    def test_different_prompt_different_key(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        k2 = AsyncLRUCache.make_key("gpt-4", "world", 0.7, 100)
        assert k1 != k2

    def test_different_temperature_different_key(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        k2 = AsyncLRUCache.make_key("gpt-4", "hello", 0.9, 100)
        assert k1 != k2

    def test_different_max_tokens_different_key(self):
        from synapsekit.llm._cache import AsyncLRUCache

        k1 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 100)
        k2 = AsyncLRUCache.make_key("gpt-4", "hello", 0.7, 200)
        assert k1 != k2

    def test_list_messages_input(self):
        from synapsekit.llm._cache import AsyncLRUCache

        messages = [{"role": "user", "content": "hi"}]
        key = AsyncLRUCache.make_key("gpt-4", messages, 0.7, 100)
        assert isinstance(key, str)
        assert len(key) in (32, 64)

    def test_complex_messages_deterministic(self):
        from synapsekit.llm._cache import AsyncLRUCache

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        k1 = AsyncLRUCache.make_key("gpt-4", messages, 0.0, 50)
        k2 = AsyncLRUCache.make_key("gpt-4", messages, 0.0, 50)
        assert k1 == k2

    def test_unicode_prompt(self):
        from synapsekit.llm._cache import AsyncLRUCache

        key = AsyncLRUCache.make_key("gpt-4", "\u4f60\u597d\u4e16\u754c\U0001f600", 0.5, 100)
        assert isinstance(key, str)
        assert len(key) in (32, 64)

    def test_empty_prompt(self):
        from synapsekit.llm._cache import AsyncLRUCache

        key = AsyncLRUCache.make_key("gpt-4", "", 0.7, 100)
        assert isinstance(key, str)

    def test_filesystem_cache_shares_make_key(self):
        """FilesystemLLMCache.make_key must produce same keys as AsyncLRUCache."""
        from synapsekit.llm._cache import AsyncLRUCache
        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        k1 = AsyncLRUCache.make_key("gpt-4", "test", 0.7, 100)
        k2 = FilesystemLLMCache.make_key("gpt-4", "test", 0.7, 100)
        assert k1 == k2


# ═══════════════════════════════════════════════════════════════════════════
# 4. __slots__ verification
# ═══════════════════════════════════════════════════════════════════════════


class TestSlots:
    def test_async_lru_cache_has_slots(self):
        from synapsekit.llm._cache import AsyncLRUCache

        assert hasattr(AsyncLRUCache, "__slots__")
        cache = AsyncLRUCache()
        with pytest.raises(AttributeError):
            cache.nonexistent_attr = 42  # type: ignore[attr-defined]

    def test_filesystem_cache_has_slots(self):
        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        assert hasattr(FilesystemLLMCache, "__slots__")

    def test_recursive_splitter_has_slots(self):
        from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter

        assert hasattr(RecursiveCharacterTextSplitter, "__slots__")
        assert "chunk_size" in RecursiveCharacterTextSplitter.__slots__

    def test_character_splitter_has_slots(self):
        from synapsekit.text_splitters.character import CharacterTextSplitter

        assert hasattr(CharacterTextSplitter, "__slots__")
        assert "separator" in CharacterTextSplitter.__slots__

    def test_json_parser_has_slots(self):
        from synapsekit.parsers.json_parser import JSONParser

        assert hasattr(JSONParser, "__slots__")
        parser = JSONParser()
        with pytest.raises(AttributeError):
            parser.nonexistent = True  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════
# 5. AsyncLRUCache — full branch coverage with __slots__
# ═══════════════════════════════════════════════════════════════════════════


class TestAsyncLRUCacheWithSlots:
    def test_put_and_get(self):
        from synapsekit.llm._cache import AsyncLRUCache

        cache = AsyncLRUCache(maxsize=10)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.hits == 1

    def test_get_miss(self):
        from synapsekit.llm._cache import AsyncLRUCache

        cache = AsyncLRUCache()
        assert cache.get("nonexistent") is None
        assert cache.misses == 1

    def test_eviction(self):
        from synapsekit.llm._cache import AsyncLRUCache

        cache = AsyncLRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_ordering(self):
        from synapsekit.llm._cache import AsyncLRUCache

        cache = AsyncLRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # touch "a", making "b" LRU
        cache.put("c", 3)  # evicts "b"
        assert cache.get("b") is None
        assert cache.get("a") == 1

    def test_update_existing_key(self):
        from synapsekit.llm._cache import AsyncLRUCache

        cache = AsyncLRUCache()
        cache.put("a", 1)
        cache.put("a", 2)
        assert cache.get("a") == 2

    def test_clear(self):
        from synapsekit.llm._cache import AsyncLRUCache

        cache = AsyncLRUCache()
        cache.put("a", 1)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_len(self):
        from synapsekit.llm._cache import AsyncLRUCache

        cache = AsyncLRUCache()
        assert len(cache) == 0
        cache.put("a", 1)
        assert len(cache) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 6. FilesystemLLMCache — JSON wiring
# ═══════════════════════════════════════════════════════════════════════════


class TestFilesystemCacheJsonWiring:
    def test_put_get_roundtrip(self, tmp_path):
        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        cache = FilesystemLLMCache(str(tmp_path / "cache"))
        cache.put("key1", {"result": "hello", "tokens": 42})
        result = cache.get("key1")
        assert result == {"result": "hello", "tokens": 42}
        assert isinstance(result["tokens"], int)

    def test_get_miss(self, tmp_path):
        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        cache = FilesystemLLMCache(str(tmp_path / "cache"))
        assert cache.get("nonexistent") is None
        assert cache.misses == 1

    def test_clear(self, tmp_path):
        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        cache = FilesystemLLMCache(str(tmp_path / "cache"))
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0

    def test_unicode_value(self, tmp_path):
        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        cache = FilesystemLLMCache(str(tmp_path / "cache"))
        cache.put("key", {"text": "\u4f60\u597d\U0001f600"})
        assert cache.get("key") == {"text": "\u4f60\u597d\U0001f600"}

    def test_nested_value(self, tmp_path):
        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        cache = FilesystemLLMCache(str(tmp_path / "cache"))
        value = {"messages": [{"role": "user", "content": "hi"}], "meta": {"tokens": 5}}
        cache.put("key", value)
        assert cache.get("key") == value


# ═══════════════════════════════════════════════════════════════════════════
# 7. VectorStore — pre-allocated buffer + vectorised MMR
# ═══════════════════════════════════════════════════════════════════════════


def _make_mock_embeddings(dim: int = 8):
    """Create a mock embedding backend that returns random unit vectors."""
    mock = AsyncMock()
    mock.embed = AsyncMock(
        side_effect=lambda texts: _random_unit_vectors(len(texts), dim)
    )
    mock.embed_one = AsyncMock(
        side_effect=lambda _text: _random_unit_vectors(1, dim)[0]
    )
    return mock


def _random_unit_vectors(n: int, dim: int) -> np.ndarray:
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


class TestVectorStoreBuffer:
    """Test pre-allocated doubling buffer behavior."""

    async def test_add_queues_pending(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        assert store._vectors is None
        await store.add(["hello"])
        assert len(store._pending) == 1
        assert store._vectors is None  # not consolidated yet

    async def test_buffer_grows_on_demand(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        # Add enough to trigger buffer growth
        for i in range(300):
            await store.add([f"doc {i}"])
        # Force consolidation
        store._consolidate()
        assert store._consolidated == 300
        assert store._buf is not None
        assert store._buf.shape[0] >= 300

    async def test_consolidate_merges_pending(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        await store.add(["a", "b"])
        await store.add(["c"])
        assert len(store._pending) == 2
        assert store._vectors is None  # not consolidated yet
        store._consolidate()
        assert len(store._pending) == 0
        assert store._vectors is not None
        assert store._vectors.shape[0] == 3

    async def test_search_triggers_consolidation(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        await store.add(["hello", "world"])
        assert len(store._pending) == 1
        results = await store.search("query", top_k=2)
        assert len(store._pending) == 0
        assert len(results) == 2

    async def test_multiple_add_batches(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        await store.add(["a"])
        await store.add(["b"])
        await store.add(["c"])
        results = await store.search("query", top_k=10)
        assert len(results) == 3
        texts = {r["text"] for r in results}
        assert texts == {"a", "b", "c"}

    async def test_empty_store_search(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        results = await store.search("query")
        assert results == []

    async def test_empty_add_is_noop(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        await store.add([])
        assert len(store) == 0

    async def test_save_load_roundtrip(self, tmp_path):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings(dim=4))
        # Use deterministic embeddings for save/load test
        vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        store._embeddings.embed = AsyncMock(return_value=vecs)
        await store.add(["doc1", "doc2"], [{"src": "a"}, {"src": "b"}])

        path = str(tmp_path / "store.npz")
        store.save(path)

        store2 = InMemoryVectorStore(_make_mock_embeddings(dim=4))
        store2.load(path)
        assert len(store2) == 2
        assert store2._texts == ["doc1", "doc2"]
        assert store2._metadata == [{"src": "a"}, {"src": "b"}]
        assert store2._vectors is not None
        assert store2._vectors.shape == (2, 4)
        assert len(store2._pending) == 0
        # Buffer has headroom
        assert store2._buf is not None
        assert store2._buf.shape[0] >= 2

    async def test_save_empty_raises(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        with pytest.raises(ValueError, match="empty"):
            store.save("/tmp/empty.npz")

    async def test_len(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        assert len(store) == 0
        await store.add(["a", "b", "c"])
        assert len(store) == 3


class TestVectorStoreMMR:
    """Test vectorised MMR implementation."""

    async def test_mmr_returns_results(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        await store.add([f"doc {i}" for i in range(10)])
        results = await store.search_mmr("query", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r["score"], float) for r in results)

    async def test_mmr_empty_store(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        results = await store.search_mmr("query")
        assert results == []

    async def test_mmr_top_k_exceeds_store_size(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        await store.add(["a", "b"])
        results = await store.search_mmr("query", top_k=10)
        assert len(results) == 2

    async def test_mmr_with_metadata_filter(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        await store.add(
            ["cat", "dog", "fish"],
            [{"type": "pet"}, {"type": "pet"}, {"type": "fish"}],
        )
        results = await store.search_mmr("query", top_k=5, metadata_filter={"type": "pet"})
        assert len(results) == 2
        assert all(r["metadata"]["type"] == "pet" for r in results)

    async def test_mmr_empty_filter_result(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        await store.add(["a"], [{"type": "x"}])
        results = await store.search_mmr("query", metadata_filter={"type": "nonexistent"})
        assert results == []

    async def test_mmr_diversity(self):
        """MMR with low lambda should prefer diverse results."""
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        dim = 8
        embeddings = _make_mock_embeddings(dim)
        store = InMemoryVectorStore(embeddings)

        # Create similar vectors (cluster 1) and one distinct vector
        np.random.seed(42)
        similar = np.array([[1, 0, 0, 0, 0, 0, 0, 0]] * 5, dtype=np.float32)
        distinct = np.array([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
        all_vecs = np.vstack([similar, distinct])
        norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        all_vecs = all_vecs / norms

        embeddings.embed = AsyncMock(return_value=all_vecs)
        embeddings.embed_one = AsyncMock(return_value=similar[0])
        await store.add([f"sim{i}" for i in range(5)] + ["distinct"])

        results = await store.search_mmr("query", top_k=2, lambda_mult=0.1)
        texts = [r["text"] for r in results]
        # With low lambda (diversity-heavy), the distinct doc should appear
        assert "distinct" in texts

    async def test_mmr_result_types(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        await store.add(["hello"], [{"key": "value"}])
        results = await store.search_mmr("query", top_k=1)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r["text"], str)
        assert isinstance(r["score"], float)
        assert isinstance(r["metadata"], dict)


class TestVectorStoreStress:
    """Stress tests for vector store buffer growth and search."""

    async def test_1000_documents(self):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings())
        texts = [f"document number {i}" for i in range(1000)]
        # Add in batches of varying sizes
        for batch_start in range(0, 1000, 50):
            await store.add(texts[batch_start : batch_start + 50])

        assert len(store) == 1000
        results = await store.search("query", top_k=10)
        assert len(results) == 10
        assert all(isinstance(r["score"], float) for r in results)

    async def test_repeated_save_load(self, tmp_path):
        from synapsekit.retrieval.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore(_make_mock_embeddings(dim=4))
        vecs = _random_unit_vectors(10, 4)
        store._embeddings.embed = AsyncMock(return_value=vecs)
        await store.add([f"d{i}" for i in range(10)])

        path = str(tmp_path / "store.npz")
        store.save(path)

        # Load → save → load cycle
        store2 = InMemoryVectorStore(_make_mock_embeddings(dim=4))
        store2.load(path)
        store2.save(str(tmp_path / "store2.npz"))

        store3 = InMemoryVectorStore(_make_mock_embeddings(dim=4))
        store3.load(str(tmp_path / "store2.npz"))
        assert len(store3) == 10
        assert store3._texts == store._texts


# ═══════════════════════════════════════════════════════════════════════════
# 8. Checkpointers — JSON wiring
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckpointerJsonWiring:
    """Verify all checkpointers correctly use _json for serialization."""

    def test_json_file_checkpointer_roundtrip(self, tmp_path):
        from synapsekit.graph.checkpointers.json_file import JSONFileCheckpointer

        cp = JSONFileCheckpointer(str(tmp_path / "checkpoints"))
        state = {"messages": ["hello"], "counter": 42, "nested": {"a": [1, 2]}}
        cp.save("graph-1", 5, state)
        result = cp.load("graph-1")
        assert result is not None
        step, loaded_state = result
        assert step == 5
        assert loaded_state == state
        assert isinstance(loaded_state["counter"], int)

    def test_json_file_checkpointer_missing(self, tmp_path):
        from synapsekit.graph.checkpointers.json_file import JSONFileCheckpointer

        cp = JSONFileCheckpointer(str(tmp_path / "checkpoints"))
        assert cp.load("nonexistent") is None

    def test_json_file_checkpointer_delete(self, tmp_path):
        from synapsekit.graph.checkpointers.json_file import JSONFileCheckpointer

        cp = JSONFileCheckpointer(str(tmp_path / "checkpoints"))
        cp.save("g1", 1, {"x": 1})
        cp.delete("g1")
        assert cp.load("g1") is None

    def test_json_file_checkpointer_unicode(self, tmp_path):
        from synapsekit.graph.checkpointers.json_file import JSONFileCheckpointer

        cp = JSONFileCheckpointer(str(tmp_path / "checkpoints"))
        state = {"text": "\u4f60\u597d\U0001f600", "list": ["\u00e9", "\u00f1"]}
        cp.save("g1", 1, state)
        _, loaded = cp.load("g1")
        assert loaded == state

    def test_sqlite_checkpointer_roundtrip(self):
        from synapsekit.graph.checkpointers.sqlite import SQLiteCheckpointer

        cp = SQLiteCheckpointer(":memory:")
        state = {"messages": ["hi"], "data": {"nested": True}}
        cp.save("g1", 3, state)
        result = cp.load("g1")
        assert result is not None
        step, loaded = result
        assert step == 3
        assert loaded == state

    def test_sqlite_checkpointer_overwrite(self):
        from synapsekit.graph.checkpointers.sqlite import SQLiteCheckpointer

        cp = SQLiteCheckpointer(":memory:")
        cp.save("g1", 1, {"v": 1})
        cp.save("g1", 2, {"v": 2})
        step, state = cp.load("g1")
        assert step == 2
        assert state == {"v": 2}

    def test_sqlite_checkpointer_delete(self):
        from synapsekit.graph.checkpointers.sqlite import SQLiteCheckpointer

        cp = SQLiteCheckpointer(":memory:")
        cp.save("g1", 1, {"x": 1})
        cp.delete("g1")
        assert cp.load("g1") is None

    def test_sqlite_checkpointer_missing(self):
        from synapsekit.graph.checkpointers.sqlite import SQLiteCheckpointer

        cp = SQLiteCheckpointer(":memory:")
        assert cp.load("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════
# 9. JSONParser — _json wiring
# ═══════════════════════════════════════════════════════════════════════════


class TestJSONParserWiring:
    def test_parse_clean_json(self):
        from synapsekit.parsers.json_parser import JSONParser

        parser = JSONParser()
        result = parser.parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_surrounding_text(self):
        from synapsekit.parsers.json_parser import JSONParser

        parser = JSONParser()
        result = parser.parse('Here is the answer: {"key": "value"} hope that helps')
        assert result == {"key": "value"}

    def test_parse_array(self):
        from synapsekit.parsers.json_parser import JSONParser

        parser = JSONParser()
        result = parser.parse("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_parse_invalid_raises(self):
        from synapsekit.parsers.json_parser import JSONParser

        parser = JSONParser()
        with pytest.raises(ValueError, match="Could not parse"):
            parser.parse("no json here at all")

    def test_parse_empty_raises(self):
        from synapsekit.parsers.json_parser import JSONParser

        parser = JSONParser()
        with pytest.raises((ValueError, TypeError)):
            parser.parse("")

    def test_return_types(self):
        from synapsekit.parsers.json_parser import JSONParser

        parser = JSONParser()
        result = parser.parse('{"count": 42, "name": "test", "flag": true}')
        assert isinstance(result["count"], int)
        assert isinstance(result["name"], str)
        assert isinstance(result["flag"], bool)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Text splitters — verify Python path still correct with __slots__
# ═══════════════════════════════════════════════════════════════════════════


class TestSplittersWithSlots:
    def test_recursive_basic(self):
        from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter

        s = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        chunks = s.split("hello world")
        assert chunks == ["hello world"]

    def test_recursive_empty(self):
        from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter

        s = RecursiveCharacterTextSplitter()
        assert s.split("") == []
        assert s.split("   ") == []

    def test_recursive_long_text(self):
        from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter

        s = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "word " * 200  # 1000 chars
        chunks = s.split(text)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_recursive_100k_chars(self):
        """Stress test: 100k character document."""
        from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter

        s = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text = "This is a sentence. " * 5000  # ~100k chars
        chunks = s.split(text)
        assert len(chunks) > 10
        # All text should be represented
        total_len = sum(len(c) for c in chunks)
        assert total_len >= len(text.strip())  # overlap adds chars

    def test_character_basic(self):
        from synapsekit.text_splitters.character import CharacterTextSplitter

        s = CharacterTextSplitter(separator="\n", chunk_size=20, chunk_overlap=0)
        chunks = s.split("line1\nline2\nline3")
        assert len(chunks) >= 1

    def test_character_empty(self):
        from synapsekit.text_splitters.character import CharacterTextSplitter

        s = CharacterTextSplitter()
        assert s.split("") == []

    def test_character_no_separator(self):
        from synapsekit.text_splitters.character import CharacterTextSplitter

        s = CharacterTextSplitter(separator="\n", chunk_size=10, chunk_overlap=0)
        text = "a" * 50  # no newlines
        chunks = s.split(text)
        assert all(len(c) <= 10 for c in chunks)

    def test_split_with_metadata_preserves_slots(self):
        """split_with_metadata from BaseSplitter must work with __slots__ subclass."""
        from synapsekit.text_splitters.recursive import RecursiveCharacterTextSplitter

        s = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        result = s.split_with_metadata("hello world", {"source": "test"})
        assert len(result) == 1
        assert result[0]["text"] == "hello world"
        assert result[0]["metadata"]["source"] == "test"
        assert result[0]["metadata"]["chunk_index"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# 11. _compat lazy imports
# ═══════════════════════════════════════════════════════════════════════════


class TestCompatLazyImports:
    def test_run_sync_works(self):
        from synapsekit._compat import run_sync

        async def coro():
            return 42

        assert run_sync(coro()) == 42

    def test_run_sync_returns_correct_type(self):
        from synapsekit._compat import run_sync

        async def coro():
            return {"key": "value"}

        result = run_sync(coro())
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_run_sync_propagates_exception(self):
        from synapsekit._compat import run_sync

        async def failing():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_sync(failing())


# ═══════════════════════════════════════════════════════════════════════════
# 12. Cross-component integration: cache key → filesystem cache
# ═══════════════════════════════════════════════════════════════════════════


class TestCacheIntegration:
    def test_end_to_end_cache_put_get(self, tmp_path):
        """Full flow: make_key → put → get with _json wiring."""
        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        cache = FilesystemLLMCache(str(tmp_path / "cache"))
        key = cache.make_key("gpt-4", "What is AI?", 0.7, 200)
        value = {"text": "AI is artificial intelligence.", "usage": {"tokens": 15}}
        cache.put(key, value)

        retrieved = cache.get(key)
        assert retrieved == value
        assert isinstance(retrieved["usage"]["tokens"], int)
        assert cache.hits == 1

    def test_cache_key_stability_across_sessions(self, tmp_path):
        """Same inputs must produce same key across separate cache instances."""
        from synapsekit.llm._filesystem_cache import FilesystemLLMCache

        cache1 = FilesystemLLMCache(str(tmp_path / "c1"))
        cache2 = FilesystemLLMCache(str(tmp_path / "c2"))
        k1 = cache1.make_key("model", [{"role": "user", "content": "hi"}], 0.5, 100)
        k2 = cache2.make_key("model", [{"role": "user", "content": "hi"}], 0.5, 100)
        assert k1 == k2
