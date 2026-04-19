from __future__ import annotations

import pytest

from synapsekit.memory import AgentMemory


@pytest.mark.asyncio
async def test_store_and_recall_semantic_relevance():
    memory = AgentMemory(backend="memory", max_episodes=50)

    await memory.store(
        agent_id="u1",
        content="User prefers Python and concise answers",
        memory_type="semantic",
    )
    await memory.store(
        agent_id="u1",
        content="User primarily writes Java code",
        memory_type="semantic",
    )

    results = await memory.recall(
        agent_id="u1",
        query="What language does the user prefer for coding? Python?",
        top_k=1,
    )

    assert len(results) == 1
    assert "python" in results[0].content.lower()


@pytest.mark.asyncio
async def test_agent_id_isolation():
    memory = AgentMemory(backend="memory")

    await memory.store(agent_id="a", content="A likes Rust", memory_type="semantic")
    await memory.store(agent_id="b", content="B likes Python", memory_type="semantic")

    results_a = await memory.recall(agent_id="a", query="What does user like?", top_k=5)
    assert all(r.agent_id == "a" for r in results_a)


@pytest.mark.asyncio
async def test_ttl_expiry_filtered_from_recall():
    memory = AgentMemory(backend="memory")

    await memory.store(
        agent_id="u1",
        content="ephemeral fact",
        memory_type="episodic",
        ttl_days=0,
    )

    results = await memory.recall(agent_id="u1", query="ephemeral", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_recall_updates_access_stats():
    memory = AgentMemory(backend="memory")

    rec = await memory.store(
        agent_id="u1", content="favorite editor is vscode", memory_type="semantic"
    )
    assert rec.access_count == 0

    results = await memory.recall(agent_id="u1", query="editor", top_k=1)
    assert len(results) == 1
    assert results[0].access_count == 1


@pytest.mark.asyncio
async def test_consolidation_on_overflow_creates_semantic_and_prunes():
    memory = AgentMemory(backend="memory", max_episodes=2)

    await memory.store(agent_id="u1", content="Episode one", memory_type="episodic")
    await memory.store(agent_id="u1", content="Episode two", memory_type="episodic")
    await memory.store(agent_id="u1", content="Episode three", memory_type="episodic")

    episodic_count = await memory.count(agent_id="u1", memory_type="episodic")
    semantic_count = await memory.count(agent_id="u1", memory_type="semantic")

    assert episodic_count <= 2
    assert semantic_count >= 1


@pytest.mark.asyncio
async def test_delete_and_clear():
    memory = AgentMemory(backend="memory")

    r1 = await memory.store(agent_id="u1", content="one", memory_type="episodic")
    await memory.store(agent_id="u1", content="two", memory_type="semantic")

    deleted = await memory.delete(agent_id="u1", record_id=r1.id)
    assert deleted

    remaining = await memory.count(agent_id="u1")
    assert remaining == 1

    cleared = await memory.clear(agent_id="u1")
    assert cleared == 1
    assert await memory.count(agent_id="u1") == 0
