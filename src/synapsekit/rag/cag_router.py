from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

from ..llm.base import BaseLLM
from ..retrieval.retriever import Retriever
from ..retrieval.token_counting import TokenCounter
from .kv_cache_store import CacheKey, KVCacheStore

logger = logging.getLogger(__name__)


def model_cache_id(llm: BaseLLM) -> str:
    """Stable identity of the model *weights* used for KV-cache keying.

    A saved llama.cpp KV state is only valid for the exact weights/quantization
    it was built with, so the config name (e.g. ``"llama-3.1-8b"``) is not
    enough — two different GGUF files can share it. Incorporate the model file
    path and size so a cache built by one file is never loaded into another.
    """

    model_id = getattr(getattr(llm, "config", None), "model", "") or ""
    model_path = getattr(llm, "_model_path", None)
    if not model_path:
        return model_id
    try:
        size = os.stat(model_path).st_size
    except OSError:
        return f"{model_id}:{model_path}"
    return f"{model_id}:{model_path}:{size}"


@dataclass
class CorpusProfile:
    estimated_tokens: int
    fingerprint: str


class CorpusAnalyzer:
    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self._counter = token_counter or TokenCounter(backend="auto")

    def analyze(self, texts: list[str]) -> CorpusProfile:
        total_tokens = sum(self._counter.count_cached(t) for t in texts)

        # Order-preserving fingerprint: the KV cache is built over the corpus
        # in insertion order, so the key MUST depend on order too — otherwise a
        # reordered corpus collides on the key and loads a mismatched state.
        # Length-prefix each text so different splits can't collide.
        hasher = hashlib.sha256()
        for t in texts:
            encoded = t.encode("utf-8")
            hasher.update(len(encoded).to_bytes(8, "big"))
            hasher.update(encoded)
        fingerprint = hasher.hexdigest()

        return CorpusProfile(estimated_tokens=total_tokens, fingerprint=fingerprint)


class CAGBackend(ABC):
    @abstractmethod
    def supports(self, llm: BaseLLM) -> bool:
        pass

    @abstractmethod
    async def build_cache(self, llm: BaseLLM, corpus_text: str) -> Any:
        pass

    @abstractmethod
    def generate_with_cache(
        self, llm: BaseLLM, cache_handle: Any, query: str
    ) -> AsyncGenerator[str]:
        # An async-generator implementation (``async def`` with ``yield``)
        # satisfies this plain ``def`` returning an ``AsyncGenerator``.
        ...

    @abstractmethod
    def load_state(self, llm: BaseLLM, state_bytes: bytes) -> None:
        pass


class CAGRouter:
    """A provider-independent router wrapping a Retriever, deciding to use CAG or RAG."""

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        cag_backend: CAGBackend | None = None,
        cache_store: KVCacheStore | None = None,
        max_cag_tokens: int | None = None,
        max_cag_context_fraction: float = 0.8,
        on_cache_miss: Literal["rebuild", "rag_fallback"] = "rag_fallback",
        stable: bool = False,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._cache_store = cache_store or KVCacheStore()
        self._max_cag_tokens = max_cag_tokens
        self._max_cag_context_fraction = max_cag_context_fraction
        self._on_cache_miss = on_cache_miss
        self._stable = stable
        self._analyzer = CorpusAnalyzer(token_counter)

        # Auto-resolve cag_backend if None
        self._cag_backend: CAGBackend | None
        if cag_backend is None:
            resolved: CAGBackend | None = None
            try:
                from ..llm._llamacpp_cag_backend import LlamaCppCAGBackend

                backend = LlamaCppCAGBackend()
                if backend.supports(llm):
                    resolved = backend
            except Exception:
                resolved = None
            self._cag_backend = resolved
        else:
            self._cag_backend = cag_backend

        self._corpus_texts: list[str] = []
        self._last_route: Literal["cag", "rag"] = "rag"

    @property
    def last_route(self) -> Literal["cag", "rag"]:
        return self._last_route

    async def add(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        self._corpus_texts.extend(texts)
        await self._retriever.add(texts, metadata)

    async def add_document(self, text: str, metadata: dict | None = None) -> None:
        self._corpus_texts.append(text)
        add_doc = getattr(self._retriever, "add_document", None)
        if callable(add_doc):
            await add_doc(text, metadata=metadata)
        else:
            await self._retriever.add([text], [metadata or {}])

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
        stable: bool | None = None,
        on_cache_miss: Literal["rebuild", "rag_fallback"] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        route = await self._determine_route(stable=stable, on_cache_miss=on_cache_miss)
        self._last_route = route

        if route == "rag":
            return await self._retriever.retrieve(
                query, top_k=top_k, metadata_filter=metadata_filter
            )

        # CAG path: return all corpus texts (which forms the full context)
        return self._corpus_texts

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
        stable: bool | None = None,
        on_cache_miss: Literal["rebuild", "rag_fallback"] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        route = await self._determine_route(stable=stable, on_cache_miss=on_cache_miss)
        self._last_route = route

        if route == "rag":
            if hasattr(self._retriever, "retrieve_with_scores"):
                return await self._retriever.retrieve_with_scores(
                    query, top_k=top_k, metadata_filter=metadata_filter
                )
            texts = await self._retriever.retrieve(
                query, top_k=top_k, metadata_filter=metadata_filter
            )
            return [{"text": t, "score": None, "metadata": {}} for t in texts]

        # CAG path: return full context documents
        return [{"text": t, "score": 1.0, "metadata": {}} for t in self._corpus_texts]

    async def _determine_route(
        self,
        stable: bool | None = None,
        on_cache_miss: Literal["rebuild", "rag_fallback"] | None = None,
    ) -> Literal["cag", "rag"]:
        try:
            # 1. Is CAG backend available and supported?
            if self._cag_backend is None or not self._cag_backend.supports(self._llm):
                return "rag"

            # 2. Is corpus stable?
            is_stable = stable if stable is not None else self._stable
            if not is_stable:
                return "rag"

            if not self._corpus_texts:
                return "rag"

            # 3. Analyze corpus size/tokens
            profile = self._analyzer.analyze(self._corpus_texts)

            # Determine threshold
            if self._max_cag_tokens is not None:
                max_tokens = self._max_cag_tokens
            else:
                n_ctx = getattr(self._llm, "_n_ctx", 2048)
                max_tokens = int(self._max_cag_context_fraction * n_ctx)

            if profile.estimated_tokens > max_tokens:
                return "rag"

            # 4. Check persisted cache existence
            model_id = model_cache_id(self._llm)
            n_ctx = getattr(self._llm, "_n_ctx", 2048)
            key = CacheKey(
                corpus_fingerprint=profile.fingerprint,
                model_id=model_id,
                n_ctx=n_ctx,
            )

            loaded = self._cache_store.load(key)
            if loaded is not None:
                # Cache hit: load the state into the LLM
                self._cag_backend.load_state(self._llm, loaded[0])
                return "cag"

            # Cache miss path
            cache_miss_strategy = on_cache_miss or self._on_cache_miss
            if cache_miss_strategy == "rebuild":
                # Build the cache
                corpus_text = "\n\n".join(self._corpus_texts)
                cache_handle = await self._cag_backend.build_cache(self._llm, corpus_text)

                # Save cache to store
                # cache_handle is {"state": state, "corpus_text": corpus_text}
                meta = {k: v for k, v in cache_handle.items() if k != "state"}
                self._cache_store.save(key, cache_handle["state"], meta)

                # load the state into LLM
                self._cag_backend.load_state(self._llm, cache_handle["state"])
                return "cag"

            return "rag"
        except Exception as e:
            logger.exception("Error in CAGRouter routing decision, falling back to RAG: %s", e)
            return "rag"
