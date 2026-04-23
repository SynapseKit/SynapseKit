from __future__ import annotations

import pytest

from synapsekit.memory import AgentMemory


@pytest.mark.asyncio
async def test_sqlite_backend_store_recall_and_persist(tmp_path):
    db = str(tmp_path / "agent_memory.db")

    mem1 = AgentMemory(backend="sqlite", path=db, max_episodes=50)
    await mem1.store(agent_id="u1", content="User likes Python", memory_type="semantic")
    await mem1.store(agent_id="u1", content="User works with FastAPI", memory_type="semantic")

    mem2 = AgentMemory(backend="sqlite", path=db, max_episodes=50)
    results = await mem2.recall(agent_id="u1", query="python preference", top_k=2)

    assert len(results) == 2
    assert any("python" in r.content.lower() for r in results)


@pytest.mark.asyncio
async def test_sqlite_backend_ttl_and_clear(tmp_path):
    db = str(tmp_path / "agent_memory_ttl.db")
    mem = AgentMemory(backend="sqlite", path=db)

    await mem.store(
        agent_id="u1",
        content="short lived note",
        memory_type="episodic",
        ttl_days=0,
    )
    await mem.store(agent_id="u1", content="long lived", memory_type="semantic")

    recalled = await mem.recall(agent_id="u1", query="note", top_k=10)
    assert all("short lived" not in r.content for r in recalled)

    cleared_sem = await mem.clear(agent_id="u1", memory_type="semantic")
    assert cleared_sem == 1
    assert await mem.count(agent_id="u1") == 0
