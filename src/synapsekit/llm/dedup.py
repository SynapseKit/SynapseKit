"""
LLM Request Coalescing (Deduplication)
======================================

Wraps any ``BaseLLM`` and coalesces concurrent identical requests into a
single LLM call.  When 50 users ask the same question at the same time,
only **one** API call is made and the result is shared with all 50 callers.

This is the same pattern that CDNs and reverse-proxies use to avoid
thundering-herd problems — applied to LLM inference.

Usage
-----
::

    from synapsekit.llm import DeduplicatedLLM, OpenAILLM, LLMConfig

    base = OpenAILLM(LLMConfig(model="gpt-4o-mini", api_key="sk-..."))
    llm  = DeduplicatedLLM(base)

    # 50 concurrent calls with the same prompt → 1 real LLM call
    results = await asyncio.gather(
        *[llm.generate("What is Python?") for _ in range(50)]
    )
    # All 50 get the same answer, only 1 API call was billed.
    print(llm.stats.savings_ratio)  # → 0.98
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .base import BaseLLM, LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-flight request tracker
# ---------------------------------------------------------------------------


class _InflightRequest:
    """Tracks a single in-flight LLM call and its coalesced waiters."""

    __slots__ = ("_completed", "error", "event", "result", "waiter_count")

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.result: str | None = None
        self.error: BaseException | None = None
        self.waiter_count: int = 0
        self._completed: bool = False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CoalescingStats:
    """Tracks deduplication metrics for observability."""

    total_requests: int = 0
    """Total number of generate() calls received."""

    coalesced_requests: int = 0
    """Requests that got a free ride on an existing in-flight call."""

    actual_llm_calls: int = 0
    """Real LLM API calls made (billed calls)."""

    errors: int = 0
    """Number of LLM calls that raised an exception."""

    @property
    def savings_ratio(self) -> float:
        """Fraction of requests that were deduplicated (0.0-1.0).

        A ratio of 0.98 means 98% of requests were free rides.
        """
        if self.total_requests == 0:
            return 0.0
        return self.coalesced_requests / self.total_requests

    @property
    def cost_multiplier(self) -> float:
        """How many requests were served per actual LLM call.

        A multiplier of 50.0 means each billed call served 50 callers.
        """
        if self.actual_llm_calls == 0:
            return 0.0
        return self.total_requests / self.actual_llm_calls

    def reset(self) -> None:
        """Zero all counters."""
        self.total_requests = 0
        self.coalesced_requests = 0
        self.actual_llm_calls = 0
        self.errors = 0


# ---------------------------------------------------------------------------
# Core middleware
# ---------------------------------------------------------------------------


class DeduplicatedLLM(BaseLLM):
    """
    A ``BaseLLM`` wrapper that coalesces concurrent identical requests.

    When multiple callers call ``generate()`` (or ``generate_with_messages()``)
    with the same prompt and parameters at the same time, only **one** actual
    LLM API call is made.  All callers receive the same result.

    This is the LLM equivalent of CDN request coalescing / thundering-herd
    protection.

    Parameters
    ----------
    llm:
        Any ``BaseLLM`` instance to wrap.
    max_inflight:
        Maximum number of distinct in-flight requests to track.  When this
        limit is reached, new requests pass through without coalescing
        (safety valve to bound memory).  Default: 1000.

    Notes
    -----
    - **Streaming** (``stream()``) passes through directly — coalescing is
      only applied to ``generate()`` and ``generate_with_messages()``.
    - If the leader request (the one that actually calls the LLM) fails,
      the error is propagated to **all** coalesced waiters.
    - If the leader task is cancelled, waiters receive a ``RuntimeError``.
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        max_inflight: int = 1000,
    ) -> None:
        if max_inflight <= 0:
            raise ValueError(f"max_inflight must be > 0, got {max_inflight!r}")

        super().__init__(llm.config)

        self._llm = llm
        self._max_inflight = max_inflight
        self._inflight: dict[str, _InflightRequest] = {}
        self._lock = asyncio.Lock()
        self._stats = CoalescingStats()

    # ------------------------------------------------------------------
    # Public stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CoalescingStats:
        """Current deduplication metrics."""
        return self._stats

    @property
    def inflight_count(self) -> int:
        """Number of currently in-flight (pending) LLM calls."""
        return len(self._inflight)

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def _make_dedup_key(self, prompt_or_messages: str | list[dict[str, Any]], **kw: Any) -> str:
        """Build a deterministic hash key from prompt + model parameters."""
        parts = [
            self._llm.config.model,
            self._llm.config.system_prompt,
            str(kw.get("temperature", self._llm.config.temperature)),
            str(kw.get("max_tokens", self._llm.config.max_tokens)),
        ]
        if isinstance(prompt_or_messages, str):
            parts.append(prompt_or_messages)
        else:
            parts.append(json.dumps(prompt_or_messages, sort_keys=True))

        raw = "\x1f".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Core deduplication engine
    # ------------------------------------------------------------------

    async def _dedup_call(
        self,
        key: str,
        fn: Callable[..., Awaitable[str]],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Core coalescing logic.  Shared by ``generate()`` and
        ``generate_with_messages()``.

        1. Acquire lock, check if an in-flight request with *key* exists.
        2. If yes → attach as a waiter, release lock, wait for result.
        3. If no  → create entry, release lock, call the real LLM, broadcast.
        """
        is_leader = False
        inflight: _InflightRequest

        async with self._lock:
            self._stats.total_requests += 1

            if key in self._inflight:
                # Coalesce — attach to existing in-flight request.
                inflight = self._inflight[key]
                inflight.waiter_count += 1
                self._stats.coalesced_requests += 1
                logger.debug(
                    "DeduplicatedLLM: coalescing request (key=%.8s…, waiters=%d).",
                    key,
                    inflight.waiter_count,
                )
            else:
                if len(self._inflight) >= self._max_inflight:
                    # Safety valve — pass through without coalescing.
                    logger.warning(
                        "DeduplicatedLLM: max_inflight=%d reached, passing through.",
                        self._max_inflight,
                    )
                    self._stats.actual_llm_calls += 1
                    return await fn(*args, **kwargs)

                # Leader — create a new in-flight entry.
                inflight = _InflightRequest()
                inflight.waiter_count = 1
                self._inflight[key] = inflight
                self._stats.actual_llm_calls += 1
                is_leader = True

        if is_leader:
            try:
                result = await fn(*args, **kwargs)
                inflight.result = result
                inflight._completed = True
            except Exception as exc:
                inflight.error = exc
                self._stats.errors += 1
            finally:
                # If the leader was cancelled (CancelledError is BaseException,
                # not Exception), ensure waiters get a clear error rather than
                # silently returning None.
                if not inflight._completed and inflight.error is None:
                    inflight.error = RuntimeError(
                        "DeduplicatedLLM: leader request was cancelled "
                        "before completing the LLM call."
                    )
                inflight.event.set()
                async with self._lock:
                    self._inflight.pop(key, None)
        else:
            await inflight.event.wait()

        if inflight.error is not None:
            raise inflight.error

        return inflight.result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        """Delegate streaming to the inner LLM (no coalescing on streams)."""
        async for token in self._llm.stream(prompt, **kw):
            yield token

    async def generate(self, prompt: str, **kw: Any) -> str:
        """Generate with request coalescing."""
        key = self._make_dedup_key(prompt, **kw)
        return await self._dedup_call(key, self._llm.generate, prompt, **kw)

    async def generate_with_messages(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        """Generate from messages with request coalescing."""
        key = self._make_dedup_key(messages, **kw)
        return await self._dedup_call(key, self._llm.generate_with_messages, messages, **kw)

    # ------------------------------------------------------------------
    # Passthrough
    # ------------------------------------------------------------------

    @property
    def config(self) -> LLMConfig:  # type: ignore[override]
        return self._llm.config

    @config.setter
    def config(self, value: LLMConfig) -> None:
        if hasattr(self, "_llm"):
            self._llm.config = value
        else:
            object.__setattr__(self, "_init_config", value)

    @property
    def tokens_used(self) -> dict[str, int]:
        return self._llm.tokens_used

    @property
    def supports_multimodal(self) -> bool:  # type: ignore[override]
        return bool(getattr(self._llm, "supports_multimodal", False))

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str]:
        async for token in self._llm.stream_with_messages(messages, **kw):
            yield token

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return await self._llm.call_with_tools(messages, tools)

    async def aclose(self) -> None:
        await self._llm.aclose()

    def __repr__(self) -> str:
        return (
            f"DeduplicatedLLM("
            f"llm={self._llm!r}, "
            f"max_inflight={self._max_inflight}, "
            f"inflight={len(self._inflight)}"
            f")"
        )
