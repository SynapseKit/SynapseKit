from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.llamacpp import LlamaCppLLM
from synapsekit.rag.cag_router import CAGRouter
from synapsekit.rag.kv_cache_store import KVCacheStore
from synapsekit.retrieval.token_counting import TokenCounter


def make_config() -> LLMConfig:
    return LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp")


@pytest.fixture
def token_counter() -> TokenCounter:
    # Use character-based counter for deterministic testing: 1 char = 1 token
    return TokenCounter(count_fn=len)


@pytest.fixture
def llm() -> LlamaCppLLM:
    instance = LlamaCppLLM(make_config(), model_path="/models/test.gguf")
    instance._model = MagicMock()
    return instance


@pytest.fixture
def retriever() -> MagicMock:
    r = MagicMock()
    r.retrieve = AsyncMock(return_value=["retrieved chunk"])
    r.retrieve_with_scores = AsyncMock(
        return_value=[{"text": "retrieved chunk", "score": 0.9, "metadata": {}}]
    )
    r.add = AsyncMock()
    return r


@pytest.fixture
def cache_store() -> MagicMock:
    store = MagicMock(spec=KVCacheStore)
    store.load.return_value = None
    return store


@pytest.mark.asyncio
async def test_router_cag_route_when_supported_stable_and_cached(
    retriever: MagicMock, llm: LlamaCppLLM, cache_store: MagicMock, token_counter: TokenCounter
) -> None:
    # Setup mock backend and cache hit
    backend = MagicMock()
    backend.supports.return_value = True
    cache_store.load.return_value = (b"state_bytes", {"metadata": {}})

    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        cag_backend=backend,
        cache_store=cache_store,
        max_cag_tokens=100,
        stable=True,
        token_counter=token_counter,
    )

    await router.add(["corpus content"])
    res = await router.retrieve("query")

    assert router.last_route == "cag"
    assert res == ["corpus content"]
    # Verify no RAG retrieval was triggered
    retriever.retrieve.assert_not_called()
    backend.load_state.assert_called_once_with(llm, b"state_bytes")


@pytest.mark.asyncio
async def test_router_rag_route_when_unstable(
    retriever: MagicMock, llm: LlamaCppLLM, cache_store: MagicMock, token_counter: TokenCounter
) -> None:
    backend = MagicMock()
    backend.supports.return_value = True

    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        cag_backend=backend,
        cache_store=cache_store,
        max_cag_tokens=100,
        stable=False,  # stable = False -> RAG
        token_counter=token_counter,
    )

    await router.add(["corpus content"])
    res = await router.retrieve("query")

    assert router.last_route == "rag"
    assert res == ["retrieved chunk"]
    retriever.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_router_rag_route_when_unsupported_backend(
    retriever: MagicMock, llm: LlamaCppLLM, cache_store: MagicMock, token_counter: TokenCounter
) -> None:
    backend = MagicMock()
    backend.supports.return_value = False  # unsupported -> RAG

    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        cag_backend=backend,
        cache_store=cache_store,
        max_cag_tokens=100,
        stable=True,
        token_counter=token_counter,
    )

    await router.add(["corpus content"])
    res = await router.retrieve("query")

    assert router.last_route == "rag"
    assert res == ["retrieved chunk"]
    retriever.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_router_rag_route_when_exceeds_threshold(
    retriever: MagicMock, llm: LlamaCppLLM, cache_store: MagicMock, token_counter: TokenCounter
) -> None:
    backend = MagicMock()
    backend.supports.return_value = True

    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        cag_backend=backend,
        cache_store=cache_store,
        max_cag_tokens=10,  # Max 10 tokens
        stable=True,
        token_counter=token_counter,
    )

    # Ingest 15 characters (which equals 15 tokens under character token_counter)
    await router.add(["fifteen chars!!"])
    res = await router.retrieve("query")

    assert router.last_route == "rag"
    assert res == ["retrieved chunk"]
    retriever.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_router_cache_miss_rebuild(
    retriever: MagicMock, llm: LlamaCppLLM, cache_store: MagicMock, token_counter: TokenCounter
) -> None:
    backend = MagicMock()
    backend.supports.return_value = True
    backend.build_cache = AsyncMock(
        return_value={"state": b"new_state", "corpus_text": "corpus content"}
    )
    cache_store.load.return_value = None  # Cache miss

    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        cag_backend=backend,
        cache_store=cache_store,
        max_cag_tokens=100,
        stable=True,
        on_cache_miss="rebuild",  # rebuild cache now
        token_counter=token_counter,
    )

    await router.add(["corpus content"])
    res = await router.retrieve("query")

    assert router.last_route == "cag"
    assert res == ["corpus content"]
    backend.build_cache.assert_awaited_once_with(llm, "corpus content")
    cache_store.save.assert_called_once()
    backend.load_state.assert_called_once_with(llm, b"new_state")


@pytest.mark.asyncio
async def test_router_cache_miss_fallback(
    retriever: MagicMock, llm: LlamaCppLLM, cache_store: MagicMock, token_counter: TokenCounter
) -> None:
    backend = MagicMock()
    backend.supports.return_value = True
    cache_store.load.return_value = None  # Cache miss

    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        cag_backend=backend,
        cache_store=cache_store,
        max_cag_tokens=100,
        stable=True,
        on_cache_miss="rag_fallback",  # fallback to RAG on miss
        token_counter=token_counter,
    )

    await router.add(["corpus content"])
    res = await router.retrieve("query")

    assert router.last_route == "rag"
    assert res == ["retrieved chunk"]
    retriever.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_router_call_level_overrides(
    retriever: MagicMock, llm: LlamaCppLLM, cache_store: MagicMock, token_counter: TokenCounter
) -> None:
    backend = MagicMock()
    backend.supports.return_value = True
    cache_store.load.return_value = None  # Cache miss

    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        cag_backend=backend,
        cache_store=cache_store,
        max_cag_tokens=100,
        stable=False,  # stable = False by default
        on_cache_miss="rag_fallback",
        token_counter=token_counter,
    )

    await router.add(["corpus content"])

    # Override stable=True and on_cache_miss="rebuild" at call level
    backend.build_cache = AsyncMock(
        return_value={"state": b"new_state", "corpus_text": "corpus content"}
    )
    res = await router.retrieve("query", stable=True, on_cache_miss="rebuild")

    assert router.last_route == "cag"
    assert res == ["corpus content"]


@pytest.mark.asyncio
async def test_router_handles_exception_and_falls_back_to_rag(
    retriever: MagicMock, llm: LlamaCppLLM, cache_store: MagicMock, token_counter: TokenCounter
) -> None:
    backend = MagicMock()
    backend.supports.side_effect = RuntimeError("Something exploded")

    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        cag_backend=backend,
        cache_store=cache_store,
        max_cag_tokens=100,
        stable=True,
        token_counter=token_counter,
    )

    await router.add(["corpus content"])
    res = await router.retrieve("query")

    assert router.last_route == "rag"
    assert res == ["retrieved chunk"]
