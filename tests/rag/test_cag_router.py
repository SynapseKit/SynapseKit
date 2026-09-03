from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from synapsekit.llm.base import LLMConfig
from synapsekit.llm.llamacpp import LlamaCppLLM
from synapsekit.rag.cag_router import CAGBackend, CAGRouter
from synapsekit.rag.kv_cache_store import CacheKey, KVCacheStore
from synapsekit.retrieval.token_counting import TokenCounter


def make_config() -> LLMConfig:
    return LLMConfig(model="llama-3.1-8b", api_key="", provider="llamacpp")


class FakeRetriever:
    """Hand-written retriever that records calls and returns fixed results."""

    def __init__(self) -> None:
        self.added: list[tuple[list[str], Any]] = []
        self.retrieve_calls = 0

    async def add(self, texts: list[str], metadata: Any = None) -> None:
        self.added.append((texts, metadata))

    async def retrieve(self, query: str, top_k: int = 5, metadata_filter: Any = None) -> list[str]:
        self.retrieve_calls += 1
        return ["retrieved chunk"]

    async def retrieve_with_scores(
        self, query: str, top_k: int = 5, metadata_filter: Any = None
    ) -> list[dict[str, Any]]:
        self.retrieve_calls += 1
        return [{"text": "retrieved chunk", "score": 0.9, "metadata": {}}]


class FakeCAGBackend(CAGBackend):
    """Hand-written CAG backend with configurable support and build results."""

    def __init__(
        self,
        *,
        supported: bool = True,
        supports_error: Exception | None = None,
        build_result: dict[str, Any] | None = None,
    ) -> None:
        self._supported = supported
        self._supports_error = supports_error
        self._build_result = build_result
        self.load_state_calls: list[tuple[Any, bytes]] = []
        self.build_cache_calls: list[tuple[Any, str]] = []

    def supports(self, llm: Any) -> bool:
        if self._supports_error is not None:
            raise self._supports_error
        return self._supported

    async def build_cache(self, llm: Any, corpus_text: str) -> Any:
        self.build_cache_calls.append((llm, corpus_text))
        return self._build_result

    async def generate_with_cache(
        self, llm: Any, cache_handle: Any, query: str
    ) -> AsyncGenerator[str]:
        # Unused by these routing tests; present to satisfy the contract.
        for chunk in ():
            yield chunk

    def load_state(self, llm: Any, state_bytes: bytes) -> None:
        self.load_state_calls.append((llm, state_bytes))


@pytest.fixture
def token_counter() -> TokenCounter:
    # Character-based counter for deterministic testing: 1 char == 1 token.
    return TokenCounter(count_fn=len)


@pytest.fixture
def llm() -> LlamaCppLLM:
    return LlamaCppLLM(make_config(), model_path="/models/test.gguf")


@pytest.fixture
def retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture
def cache_store(tmp_path) -> KVCacheStore:
    return KVCacheStore(cache_dir=str(tmp_path / "cag_cache"))


def _cache_key(llm: LlamaCppLLM, texts: list[str]) -> CacheKey:
    """Recompute the store key the router derives for ``texts``."""
    hasher = hashlib.sha256()
    for text in sorted(texts):
        hasher.update(text.encode("utf-8"))
    return CacheKey(
        corpus_fingerprint=hasher.hexdigest(),
        model_id=llm.config.model,
        n_ctx=getattr(llm, "_n_ctx", 2048),
    )


@pytest.mark.asyncio
async def test_router_cag_route_when_supported_stable_and_cached(
    retriever: FakeRetriever,
    llm: LlamaCppLLM,
    cache_store: KVCacheStore,
    token_counter: TokenCounter,
) -> None:
    backend = FakeCAGBackend(supported=True)
    # Seed the real cache store so the routing decision finds a hit.
    cache_store.save(_cache_key(llm, ["corpus content"]), b"state_bytes", {"metadata": {}})

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
    # No RAG retrieval was triggered, and the cached state was loaded.
    assert retriever.retrieve_calls == 0
    assert backend.load_state_calls == [(llm, b"state_bytes")]


@pytest.mark.asyncio
async def test_router_rag_route_when_unstable(
    retriever: FakeRetriever,
    llm: LlamaCppLLM,
    cache_store: KVCacheStore,
    token_counter: TokenCounter,
) -> None:
    backend = FakeCAGBackend(supported=True)

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
    assert retriever.retrieve_calls == 1


@pytest.mark.asyncio
async def test_router_rag_route_when_unsupported_backend(
    retriever: FakeRetriever,
    llm: LlamaCppLLM,
    cache_store: KVCacheStore,
    token_counter: TokenCounter,
) -> None:
    backend = FakeCAGBackend(supported=False)  # unsupported -> RAG

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
    assert retriever.retrieve_calls == 1


@pytest.mark.asyncio
async def test_router_rag_route_when_exceeds_threshold(
    retriever: FakeRetriever,
    llm: LlamaCppLLM,
    cache_store: KVCacheStore,
    token_counter: TokenCounter,
) -> None:
    backend = FakeCAGBackend(supported=True)

    router = CAGRouter(
        retriever=retriever,
        llm=llm,
        cag_backend=backend,
        cache_store=cache_store,
        max_cag_tokens=10,  # Max 10 tokens
        stable=True,
        token_counter=token_counter,
    )

    # 15 characters == 15 tokens under the character token_counter, over the cap.
    await router.add(["fifteen chars!!"])
    res = await router.retrieve("query")

    assert router.last_route == "rag"
    assert res == ["retrieved chunk"]
    assert retriever.retrieve_calls == 1


@pytest.mark.asyncio
async def test_router_cache_miss_rebuild(
    retriever: FakeRetriever,
    llm: LlamaCppLLM,
    cache_store: KVCacheStore,
    token_counter: TokenCounter,
) -> None:
    backend = FakeCAGBackend(
        supported=True,
        build_result={"state": b"new_state", "corpus_text": "corpus content"},
    )

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
    assert backend.build_cache_calls == [(llm, "corpus content")]
    assert backend.load_state_calls == [(llm, b"new_state")]
    # The rebuilt cache was persisted to the real store and reloads.
    loaded = cache_store.load(_cache_key(llm, ["corpus content"]))
    assert loaded is not None
    assert loaded[0] == b"new_state"


@pytest.mark.asyncio
async def test_router_cache_miss_fallback(
    retriever: FakeRetriever,
    llm: LlamaCppLLM,
    cache_store: KVCacheStore,
    token_counter: TokenCounter,
) -> None:
    backend = FakeCAGBackend(supported=True)

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
    assert retriever.retrieve_calls == 1


@pytest.mark.asyncio
async def test_router_call_level_overrides(
    retriever: FakeRetriever,
    llm: LlamaCppLLM,
    cache_store: KVCacheStore,
    token_counter: TokenCounter,
) -> None:
    backend = FakeCAGBackend(
        supported=True,
        build_result={"state": b"new_state", "corpus_text": "corpus content"},
    )

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

    # Override stable=True and on_cache_miss="rebuild" at call level.
    res = await router.retrieve("query", stable=True, on_cache_miss="rebuild")

    assert router.last_route == "cag"
    assert res == ["corpus content"]


@pytest.mark.asyncio
async def test_router_handles_exception_and_falls_back_to_rag(
    retriever: FakeRetriever,
    llm: LlamaCppLLM,
    cache_store: KVCacheStore,
    token_counter: TokenCounter,
) -> None:
    backend = FakeCAGBackend(supports_error=RuntimeError("Something exploded"))

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
    assert retriever.retrieve_calls == 1
