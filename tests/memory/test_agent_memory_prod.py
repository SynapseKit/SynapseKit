"""Production-grade tests for AgentMemory — covers async embedder, cosine edge cases,
Redis/Postgres backends (mocked), consolidation with LLM, count/delete, unknown backend,
and integration edge cases."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.memory import AgentMemory
from synapsekit.memory.agent_memory import AgentMemory as _AgentMemory
from synapsekit.memory.backends.memory import InMemoryMemoryBackend
from synapsekit.memory.base import MemoryRecord, MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    agent_id: str = "agent-1",
    content: str = "hello",
    memory_type: MemoryType = "episodic",
    ttl_days: int | None = None,
    access_count: int = 0,
    age_days: float = 0.0,
) -> MemoryRecord:
    now = datetime.now(timezone.utc) - timedelta(days=age_days)
    return MemoryRecord(
        id=f"rec-{abs(hash(content + str(age_days))) % 99999}",
        agent_id=agent_id,
        content=content,
        memory_type=memory_type,
        embedding=[0.1, 0.9] + [0.0] * 126,
        created_at=now,
        accessed_at=now,
        access_count=access_count,
        ttl_days=ttl_days,
        metadata={},
    )


# ---------------------------------------------------------------------------
# 1. _default_embed edge cases
# ---------------------------------------------------------------------------


def test_default_embed_empty_text_returns_zeros():
    mem = _AgentMemory(backend="memory")
    result = mem._default_embed("")
    assert len(result) == 128
    assert all(v == 0.0 for v in result)


def test_default_embed_nonempty_is_unit_normalised():
    import math

    mem = _AgentMemory(backend="memory")
    vec = mem._default_embed("the quick brown fox")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. _embed_text with sync and async embedders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_text_uses_async_embedder():
    async def async_embedder(text: str) -> list[float]:
        return [1.0] + [0.0] * 127

    mem = _AgentMemory(backend="memory", embedder=async_embedder)
    result = await mem._embed_text("anything")
    assert result[0] == pytest.approx(1.0)
    assert len(result) == 128


@pytest.mark.asyncio
async def test_embed_text_uses_sync_embedder():
    called_with: list[str] = []

    def sync_embedder(text: str) -> list[float]:
        called_with.append(text)
        return [0.5] * 128

    mem = _AgentMemory(backend="memory", embedder=sync_embedder)
    result = await mem._embed_text("hello world")
    assert called_with == ["hello world"]
    assert result == [0.5] * 128


# ---------------------------------------------------------------------------
# 3. _cosine edge cases
# ---------------------------------------------------------------------------


def test_cosine_empty_vectors_returns_zero():
    assert _AgentMemory._cosine([], []) == 0.0


def test_cosine_mismatched_lengths_returns_zero():
    assert _AgentMemory._cosine([1.0, 0.0], [1.0]) == 0.0


def test_cosine_zero_norm_returns_zero():
    assert _AgentMemory._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_orthogonal_vectors():
    assert _AgentMemory._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_identical_vectors():
    assert _AgentMemory._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Backend resolution
# ---------------------------------------------------------------------------


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        _AgentMemory(backend="nonexistent_db")


def test_postgres_backend_requires_dsn():
    with pytest.raises(ValueError, match="postgres_dsn"):
        _AgentMemory(backend="postgres")


def test_custom_backend_instance_accepted():
    custom = InMemoryMemoryBackend()
    mem = _AgentMemory(backend=custom)
    assert mem._backend is custom


# ---------------------------------------------------------------------------
# 5. store → recall round-trip (in-memory backend)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_recall_basic():
    mem = AgentMemory(backend="memory")
    await mem.store(
        agent_id="user-1", content="Python was created by Guido van Rossum", memory_type="semantic"
    )
    results = await mem.recall(agent_id="user-1", query="who created Python")
    assert len(results) >= 1
    assert any("Guido" in r.content for r in results)


@pytest.mark.asyncio
async def test_recall_top_k_zero_returns_empty():
    mem = AgentMemory(backend="memory")
    await mem.store(agent_id="user-1", content="something", memory_type="episodic")
    results = await mem.recall(agent_id="user-1", query="something", top_k=0)
    assert results == []


@pytest.mark.asyncio
async def test_recall_empty_store_returns_empty():
    mem = AgentMemory(backend="memory")
    results = await mem.recall(agent_id="user-1", query="query")
    assert results == []


@pytest.mark.asyncio
async def test_recall_filters_by_memory_type():
    mem = AgentMemory(backend="memory")
    await mem.store(agent_id="u1", content="episodic fact", memory_type="episodic")
    await mem.store(agent_id="u1", content="semantic fact", memory_type="semantic")

    only_episodic = await mem.recall(agent_id="u1", query="fact", memory_types=("episodic",))
    assert all(r.memory_type == "episodic" for r in only_episodic)

    only_semantic = await mem.recall(agent_id="u1", query="fact", memory_types=("semantic",))
    assert all(r.memory_type == "semantic" for r in only_semantic)


@pytest.mark.asyncio
async def test_recall_increments_access_count():
    mem = AgentMemory(backend="memory")
    await mem.store(agent_id="u1", content="unique access content xkcd", memory_type="semantic")

    before = await mem.recall(agent_id="u1", query="unique access content xkcd", top_k=1)
    first_count = before[0].access_count

    await mem.recall(agent_id="u1", query="unique access content xkcd", top_k=1)
    after = await mem.recall(agent_id="u1", query="unique access content xkcd", top_k=1)
    assert after[0].access_count > first_count


# ---------------------------------------------------------------------------
# 6. count + delete + clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_returns_correct_total():
    mem = AgentMemory(backend="memory")
    assert await mem.count(agent_id="u1") == 0
    await mem.store(agent_id="u1", content="a", memory_type="episodic")
    await mem.store(agent_id="u1", content="b", memory_type="episodic")
    await mem.store(agent_id="u1", content="c", memory_type="semantic")
    assert await mem.count(agent_id="u1") == 3
    assert await mem.count(agent_id="u1", memory_type="episodic") == 2
    assert await mem.count(agent_id="u1", memory_type="semantic") == 1


@pytest.mark.asyncio
async def test_delete_removes_specific_record():
    mem = AgentMemory(backend="memory")
    await mem.store(agent_id="u1", content="keep me", memory_type="episodic")
    records = await mem.recall(agent_id="u1", query="keep me", top_k=1)
    rid = records[0].id

    await mem.delete(agent_id="u1", record_id=rid)
    after = await mem.recall(agent_id="u1", query="keep me", top_k=5)
    assert all(r.id != rid for r in after)


@pytest.mark.asyncio
async def test_clear_wipes_agent_memory():
    mem = AgentMemory(backend="memory")
    await mem.store(agent_id="u1", content="will be cleared", memory_type="episodic")
    await mem.store(agent_id="u1", content="also cleared", memory_type="semantic")
    await mem.clear(agent_id="u1")
    assert await mem.count(agent_id="u1") == 0


# ---------------------------------------------------------------------------
# 7. TTL expiry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_records_pruned_before_recall():
    mem = AgentMemory(backend="memory")
    await mem.store(agent_id="u1", content="short lived", memory_type="episodic", ttl_days=0)
    results = await mem.recall(agent_id="u1", query="short lived", top_k=5)
    assert all("short lived" not in r.content for r in results)


# ---------------------------------------------------------------------------
# 8. Consolidation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidate_empty_returns_none():
    mem = AgentMemory(backend="memory")
    result = await mem.consolidate(agent_id="nobody")
    assert result is None


@pytest.mark.asyncio
async def test_consolidate_with_llm_called():
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="consolidated summary bullet points")

    mem = AgentMemory(backend="memory", llm=mock_llm, max_episodes=3, consolidation_window=3)
    for i in range(4):
        await mem.store(agent_id="u1", content=f"episode {i}", memory_type="episodic")

    result = await mem.consolidate(agent_id="u1")
    assert result is not None
    # auto-consolidation may fire during store() calls; verify at least one LLM call occurred
    assert mock_llm.generate.call_count >= 1


@pytest.mark.asyncio
async def test_consolidate_fallback_when_llm_raises():
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

    mem = AgentMemory(backend="memory", llm=mock_llm, max_episodes=2, consolidation_window=2)
    await mem.store(agent_id="u1", content="episode alpha", memory_type="episodic")
    await mem.store(agent_id="u1", content="episode beta", memory_type="episodic")

    result = await mem.consolidate(agent_id="u1")
    assert result is not None
    assert result.memory_type == "semantic"


@pytest.mark.asyncio
async def test_consolidate_explicit_limit():
    mem = AgentMemory(backend="memory")
    for i in range(5):
        await mem.store(agent_id="u1", content=f"ep {i}", memory_type="episodic")

    result = await mem.consolidate(agent_id="u1", limit=3)
    assert result is not None
    assert result.memory_type == "semantic"


# ---------------------------------------------------------------------------
# 9. InMemoryMemoryBackend — full branch coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_backend_touch_updates_accessed_at():
    backend = InMemoryMemoryBackend()
    rec = _make_record(agent_id="a", content="hello", age_days=1.0)
    original_accessed = rec.accessed_at
    await backend.store(rec)

    await backend.touch("a", rec.id)
    fetched = await backend.fetch("a")
    assert fetched[0].accessed_at >= original_accessed


@pytest.mark.asyncio
async def test_inmemory_backend_fetch_filters_by_type():
    backend = InMemoryMemoryBackend()
    await backend.store(_make_record("a", "ep1", "episodic"))
    await backend.store(_make_record("a", "sem1", "semantic"))

    episodic = await backend.fetch("a", memory_type="episodic")
    semantic = await backend.fetch("a", memory_type="semantic")
    assert len(episodic) == 1 and episodic[0].content == "ep1"
    assert len(semantic) == 1 and semantic[0].content == "sem1"


@pytest.mark.asyncio
async def test_inmemory_backend_prune_expired_removes_expired():
    backend = InMemoryMemoryBackend()
    expired = _make_record("a", "expired", ttl_days=0, age_days=2.0)
    live = _make_record("a", "live")
    await backend.store(expired)
    await backend.store(live)

    await backend.prune_expired()
    remaining = await backend.fetch("a", include_expired=True)
    contents = [r.content for r in remaining]
    assert "live" in contents
    assert "expired" not in contents


@pytest.mark.asyncio
async def test_inmemory_backend_delete_missing_id_is_noop():
    backend = InMemoryMemoryBackend()
    # Should not raise
    await backend.delete("a", "nonexistent-id")


@pytest.mark.asyncio
async def test_inmemory_backend_clear_leaves_other_agents():
    backend = InMemoryMemoryBackend()
    await backend.store(_make_record("agent-a", "A data"))
    await backend.store(_make_record("agent-b", "B data"))

    await backend.clear("agent-a")
    assert await backend.count("agent-a") == 0
    assert await backend.count("agent-b") == 1


@pytest.mark.asyncio
async def test_inmemory_backend_count_by_type():
    backend = InMemoryMemoryBackend()
    await backend.store(_make_record("a", "ep1", "episodic"))
    await backend.store(_make_record("a", "ep2", "episodic"))
    await backend.store(_make_record("a", "sem1", "semantic"))

    assert await backend.count("a") == 3
    assert await backend.count("a", memory_type="episodic") == 2
    assert await backend.count("a", memory_type="semantic") == 1


# ---------------------------------------------------------------------------
# 10. Redis backend — mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redis_backend_store_calls_pipeline():
    from synapsekit.memory.backends.redis import RedisMemoryBackend

    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.zadd = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[True, 1])

    mock_client = MagicMock()
    mock_client.pipeline.return_value = mock_pipe

    backend = RedisMemoryBackend.__new__(RedisMemoryBackend)
    backend._redis = MagicMock()
    backend._client = mock_client
    backend._prefix = "sk:"

    rec = _make_record("agent", "redis content", "episodic")
    await backend.store(rec)

    mock_client.pipeline.assert_called_once()
    mock_pipe.set.assert_called_once()
    mock_pipe.zadd.assert_called_once()
    mock_pipe.execute.assert_called_once()


@pytest.mark.asyncio
async def test_redis_backend_clear_calls_delete():
    from synapsekit.memory.backends.redis import RedisMemoryBackend

    # clear() uses pipeline: pipe.delete(*keys) + pipe.delete(index_key), then pipe.execute()
    mock_pipe = MagicMock()
    mock_pipe.delete = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[2, 1])

    mock_client = MagicMock()
    mock_client.zrange = AsyncMock(return_value=["id1", "id2"])
    mock_client.pipeline = MagicMock(return_value=mock_pipe)

    backend = RedisMemoryBackend.__new__(RedisMemoryBackend)
    backend._client = mock_client
    backend._prefix = "sk:"

    result = await backend.clear("agent-x")
    # pipeline().delete() called twice: once for all record keys, once for index key
    assert mock_pipe.delete.call_count == 2
    mock_pipe.execute.assert_called_once()
    assert result == 2  # len(ids)


@pytest.mark.asyncio
async def test_redis_backend_import_error():
    original = sys.modules.copy()
    sys.modules["redis"] = None  # type: ignore[assignment]
    sys.modules["redis.asyncio"] = None  # type: ignore[assignment]
    try:
        # Force reimport
        if "synapsekit.memory.backends.redis" in sys.modules:
            del sys.modules["synapsekit.memory.backends.redis"]
        from synapsekit.memory.backends.redis import RedisMemoryBackend

        with pytest.raises(ImportError, match="redis package required"):
            RedisMemoryBackend(url="redis://localhost:6379")
    finally:
        sys.modules.clear()
        sys.modules.update(original)


# ---------------------------------------------------------------------------
# 11. Postgres backend — mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_postgres_backend_store_uses_execute():
    from synapsekit.memory.backends.postgres import PostgresMemoryBackend

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=_AsyncCtx(mock_conn))

    backend = PostgresMemoryBackend.__new__(PostgresMemoryBackend)
    backend._pool = mock_pool
    backend._table = "agent_memory_records"

    rec = _make_record("agent-pg", "pg content", "episodic")
    await backend.store(rec)
    mock_conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_postgres_backend_fetch_empty_returns_empty_list():
    from synapsekit.memory.backends.postgres import PostgresMemoryBackend

    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=_AsyncCtx(mock_conn))

    backend = PostgresMemoryBackend.__new__(PostgresMemoryBackend)
    backend._pool = mock_pool
    backend._table = "agent_memory_records"

    result = await backend.fetch("agent-pg")
    assert result == []


# ---------------------------------------------------------------------------
# 12. MongoDBAtlasVectorStore — mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mongodb_atlas_add_and_search():
    import numpy as np

    from synapsekit.retrieval.mongodb_atlas import MongoDBAtlasVectorStore

    mock_collection = MagicMock()
    mock_collection.insert_many = MagicMock()
    mock_collection.aggregate = MagicMock(
        return_value=[{"text": "hello", "metadata": {"src": "test"}, "score": 0.95}]
    )

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)
    mock_client = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)

    mock_embeddings = AsyncMock()
    mock_embeddings.embed = AsyncMock(return_value=[np.array([0.1] * 128)])

    store = MongoDBAtlasVectorStore(
        embedding_backend=mock_embeddings,
        client=mock_client,
    )

    await store.add(["hello world"], metadata=[{"src": "test"}])
    mock_collection.insert_many.assert_called_once()

    results = await store.search("hello")
    assert len(results) == 1
    assert results[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_mongodb_atlas_search_top_k_zero():
    from synapsekit.retrieval.mongodb_atlas import MongoDBAtlasVectorStore

    mock_embeddings = AsyncMock()
    mock_client = MagicMock()
    mock_client.__getitem__ = MagicMock(
        return_value=MagicMock(__getitem__=MagicMock(return_value=MagicMock()))
    )
    store = MongoDBAtlasVectorStore(embedding_backend=mock_embeddings, client=mock_client)
    results = await store.search("query", top_k=0)
    assert results == []


@pytest.mark.asyncio
async def test_mongodb_atlas_add_empty_is_noop():
    from synapsekit.retrieval.mongodb_atlas import MongoDBAtlasVectorStore

    mock_embeddings = AsyncMock()
    mock_collection = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)
    mock_client = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)

    store = MongoDBAtlasVectorStore(embedding_backend=mock_embeddings, client=mock_client)
    await store.add([])
    mock_collection.insert_many.assert_not_called()


@pytest.mark.asyncio
async def test_mongodb_atlas_add_metadata_length_mismatch_raises():
    import numpy as np

    from synapsekit.retrieval.mongodb_atlas import MongoDBAtlasVectorStore

    mock_embeddings = AsyncMock()
    mock_embeddings.embed = AsyncMock(return_value=[np.array([0.1] * 128)])
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=MagicMock())
    mock_client = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)

    store = MongoDBAtlasVectorStore(embedding_backend=mock_embeddings, client=mock_client)
    with pytest.raises(ValueError, match="metadata must match"):
        await store.add(["text"], metadata=[{"a": 1}, {"b": 2}])


@pytest.mark.asyncio
async def test_mongodb_atlas_metadata_filter_mql_passthrough():
    """Raw MQL operators ($and/$or) are passed through unchanged."""
    import numpy as np

    from synapsekit.retrieval.mongodb_atlas import MongoDBAtlasVectorStore

    captured_pipelines: list = []
    mock_collection = MagicMock()
    mock_collection.aggregate = MagicMock(side_effect=lambda p: captured_pipelines.append(p) or [])
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)
    mock_client = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)

    mock_embeddings = AsyncMock()
    mock_embeddings.embed = AsyncMock(return_value=[np.array([0.1] * 128)])

    store = MongoDBAtlasVectorStore(embedding_backend=mock_embeddings, client=mock_client)
    await store.search("query", metadata_filter={"$and": [{"field": "val"}]})

    assert len(captured_pipelines) == 1
    pipeline = captured_pipelines[0]
    vector_stage = pipeline[0]["$vectorSearch"]
    assert "filter" in vector_stage
    assert "$and" in vector_stage["filter"]


# ---------------------------------------------------------------------------
# 13. Custom async embedder round-trip through AgentMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_async_embedder_called_on_store_and_recall():
    calls: list[str] = []

    async def tracking_embedder(text: str) -> list[float]:
        calls.append(text)
        import hashlib

        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(h >> i & 1) * 1.0 for i in range(128)]

    mem = AgentMemory(backend="memory", embedder=tracking_embedder)
    await mem.store(agent_id="u1", content="test content", memory_type="semantic")
    await mem.recall(agent_id="u1", query="test content", top_k=1)

    assert len(calls) >= 2


# ---------------------------------------------------------------------------
# 14. ReactAgent with persistent_memory plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_agent_accepts_persistent_memory():
    from synapsekit.agents.react import ReActAgent as ReactAgent
    from synapsekit.llm.base import LLMConfig

    mem = AgentMemory(backend="memory")

    class _FakeLLM:
        config = LLMConfig(model="fake", api_key="sk-test", provider="fake")

        async def generate(self, prompt: str, **kw: object) -> str:
            return "Final Answer: done"

        async def generate_with_messages(self, messages, **kw):
            return "Final Answer: done"

        async def stream_with_messages(self, messages, **kw):
            yield "Final Answer: done"

    agent = ReactAgent(llm=_FakeLLM(), tools=[], memory=mem, agent_id="test-agent", memory_top_k=2)
    result = await agent.run("simple task")
    assert result is not None


# ---------------------------------------------------------------------------
# Utility: async context manager helper
# ---------------------------------------------------------------------------


class _AsyncCtx:
    """Minimal async context manager that yields a fixed value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *_):
        pass
