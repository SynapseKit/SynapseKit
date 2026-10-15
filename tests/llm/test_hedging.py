"""Tests for HedgedLLM and Request Hedging (p99 tail latency mitigation)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.llm.hedging import HedgedLLM, HedgingConfig, HedgingStats

# ---------------------------------------------------------------------------
# Hand-written Fake LLM fixtures (strictly zero mock libraries)
# ---------------------------------------------------------------------------


class FakeLLM(BaseLLM):
    """Controllable fake LLM for testing request hedging."""

    def __init__(
        self,
        name: str = "fake",
        response: str = "fake response",
        delay: float = 0.0,
        stream_chunks: list[str] | None = None,
        fail: bool = False,
        multimodal: bool = False,
    ) -> None:
        super().__init__(LLMConfig(model=name, api_key="test-key", provider="fake"))
        self._response = response
        self._delay = delay
        self._stream_chunks = stream_chunks if stream_chunks is not None else ["chunk1", "chunk2"]
        self._fail = fail
        self.call_count = 0
        self.closed = False
        self.supports_multimodal = multimodal

    async def generate(self, prompt: str, **kw: Any) -> str:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"{self.config.model} failed")
        return self._response

    async def generate_with_messages(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"{self.config.model} failed")
        return self._response

    async def call_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"{self.config.model} failed tool call")
        return {"tool_calls": [{"name": "fake_tool", "arguments": {}}]}

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        self.call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"{self.config.model} stream failed")
        for chunk in self._stream_chunks:
            yield chunk

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str]:
        async for chunk in self.stream("prompt", **kw):
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# HedgingConfig & Stats Tests
# ---------------------------------------------------------------------------


class TestHedgingConfigAndStats:
    def test_invalid_hedge_delay_raises_value_error(self) -> None:
        p = FakeLLM("primary")
        s = FakeLLM("secondary")
        with pytest.raises(ValueError, match="hedge_delay must be positive"):
            HedgingConfig(primary=p, secondary=s, hedge_delay=0.0)

        with pytest.raises(ValueError, match="hedge_delay must be positive"):
            HedgingConfig(primary=p, secondary=s, hedge_delay=-1.5)

    def test_stats_rates_zero_division_safe(self) -> None:
        stats = HedgingStats()
        assert stats.hedge_rate == 0.0
        assert stats.hedge_win_rate == 0.0

    def test_stats_metrics_calculation(self) -> None:
        stats = HedgingStats(
            total_requests=10,
            hedged_requests=4,
            primary_won=7,
            hedge_won=3,
        )
        assert stats.hedge_rate == 0.4
        assert stats.hedge_win_rate == 0.75


# ---------------------------------------------------------------------------
# HedgedLLM Structural & Configuration Tests
# ---------------------------------------------------------------------------


class TestHedgedLLMBasics:
    def test_drop_in_base_llm_compatibility(self) -> None:
        p = FakeLLM("p-model")
        s = FakeLLM("s-model")
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=1.0))

        assert isinstance(hedged, BaseLLM)
        assert hedged.config.model == "hedged(p-model,s-model)"
        assert hedged.config.provider == "hedged"
        assert hedged.last_winner is None

    def test_supports_multimodal(self) -> None:
        p1 = FakeLLM("p1", multimodal=False)
        s1 = FakeLLM("s1", multimodal=True)
        hedged1 = HedgedLLM(HedgingConfig(primary=p1, secondary=s1, hedge_delay=1.0))
        assert not hedged1.supports_multimodal

        p2 = FakeLLM("p2", multimodal=True)
        s2 = FakeLLM("s2", multimodal=True)
        hedged2 = HedgedLLM(HedgingConfig(primary=p2, secondary=s2, hedge_delay=1.0))
        assert hedged2.supports_multimodal

    async def test_aclose_closes_both_models(self) -> None:
        p = FakeLLM("primary")
        s = FakeLLM("secondary")
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=1.0))

        assert not p.closed
        assert not s.closed
        await hedged.aclose()
        assert p.closed
        assert s.closed


# ---------------------------------------------------------------------------
# HedgedLLM Generate Tests
# ---------------------------------------------------------------------------


class TestHedgedLLMGenerate:
    async def test_primary_wins_fast_path(self) -> None:
        p = FakeLLM("primary", response="fast primary", delay=0.01)
        s = FakeLLM("secondary", response="backup secondary", delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.1))

        res = await hedged.generate("Hello world")
        assert res == "fast primary"
        assert hedged.last_winner == "primary"
        assert hedged.stats.total_requests == 1
        assert hedged.stats.primary_won == 1
        assert hedged.stats.hedged_requests == 0
        assert hedged.stats.hedge_won == 0
        assert hedged.stats.hedge_rate == 0.0
        assert s.call_count == 0

    async def test_hedge_triggered_and_secondary_wins(self) -> None:
        p = FakeLLM("primary", response="laggy primary", delay=0.25)
        s = FakeLLM("secondary", response="speedy secondary", delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.04))

        res = await hedged.generate("Hello world")
        assert res == "speedy secondary"
        assert hedged.last_winner == "secondary"
        assert hedged.stats.total_requests == 1
        assert hedged.stats.hedged_requests == 1
        assert hedged.stats.hedge_won == 1
        assert hedged.stats.primary_won == 0
        assert hedged.stats.hedge_win_rate == 1.0
        assert s.call_count == 1

    async def test_primary_wins_even_after_hedge_triggered(self) -> None:
        # Primary takes 0.06s. Hedge delay is 0.03s.
        # Secondary is dispatched at 0.03s, but secondary takes 0.15s.
        # Primary finishes at 0.06s (earlier than secondary at 0.03 + 0.15 = 0.18s).
        p = FakeLLM("primary", response="primary comeback", delay=0.06)
        s = FakeLLM("secondary", response="slow secondary", delay=0.15)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.03))

        res = await hedged.generate("Hello world")
        assert res == "primary comeback"
        assert hedged.last_winner == "primary"
        assert hedged.stats.total_requests == 1
        assert hedged.stats.hedged_requests == 1
        assert hedged.stats.primary_won == 1
        assert hedged.stats.hedge_won == 0

    async def test_primary_fails_early_triggers_immediate_hedge(self) -> None:
        p = FakeLLM("primary", fail=True, delay=0.01)
        s = FakeLLM("secondary", response="fallback success", delay=0.02)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.5))

        # Secondary should run immediately without waiting for the full 0.5s SLA timeout
        res = await hedged.generate("Hello world")
        assert res == "fallback success"
        assert hedged.last_winner == "secondary"
        assert hedged.stats.total_requests == 1
        assert hedged.stats.hedged_requests == 1
        assert hedged.stats.hedge_won == 1

    async def test_both_fail_propagates_exception(self) -> None:
        p = FakeLLM("primary", fail=True, delay=0.01)
        s = FakeLLM("secondary", fail=True, delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.05))

        with pytest.raises(RuntimeError):
            await hedged.generate("Hello world")

    async def test_generate_with_messages(self) -> None:
        p = FakeLLM("primary", response="messages response", delay=0.01)
        s = FakeLLM("secondary", delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.1))

        res = await hedged.generate_with_messages([{"role": "user", "content": "hi"}])
        assert res == "messages response"
        assert hedged.last_winner == "primary"

    async def test_call_with_tools_hedged(self) -> None:
        p = FakeLLM("primary", delay=0.2)
        s = FakeLLM("secondary", delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.03))

        res = await hedged.call_with_tools(
            messages=[{"role": "user", "content": "run tool"}],
            tools=[{"type": "function", "function": {"name": "fake_tool"}}],
        )
        assert "tool_calls" in res
        assert hedged.last_winner == "secondary"


# ---------------------------------------------------------------------------
# HedgedLLM Streaming Tests
# ---------------------------------------------------------------------------


class TestHedgedLLMStreaming:
    async def test_streaming_primary_wins_fast_path(self) -> None:
        p = FakeLLM("primary", stream_chunks=["hello", " ", "world"], delay=0.01)
        s = FakeLLM("secondary", stream_chunks=["backup"], delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.1))

        chunks: list[str] = []
        async for chunk in hedged.stream("test prompt"):
            chunks.append(chunk)

        assert "".join(chunks) == "hello world"
        assert hedged.last_winner == "primary"
        assert hedged.stats.primary_won == 1
        assert hedged.stats.hedged_requests == 0
        assert s.call_count == 0

    async def test_streaming_hedge_triggered_and_secondary_wins(self) -> None:
        p = FakeLLM("primary", stream_chunks=["slow", "chunks"], delay=0.25)
        s = FakeLLM("secondary", stream_chunks=["fast", " ", "hedge"], delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.03))

        chunks: list[str] = []
        async for chunk in hedged.stream("test prompt"):
            chunks.append(chunk)

        assert "".join(chunks) == "fast hedge"
        assert hedged.last_winner == "secondary"
        assert hedged.stats.hedged_requests == 1
        assert hedged.stats.hedge_won == 1

    async def test_streaming_primary_fails_early_secondary_streams(self) -> None:
        p = FakeLLM("primary", fail=True, delay=0.01)
        s = FakeLLM("secondary", stream_chunks=["recovered", " ", "stream"], delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.5))

        chunks: list[str] = []
        async for chunk in hedged.stream("test prompt"):
            chunks.append(chunk)

        assert "".join(chunks) == "recovered stream"
        assert hedged.last_winner == "secondary"
        assert hedged.stats.hedge_won == 1

    async def test_streaming_both_fail_raises_exception(self) -> None:
        p = FakeLLM("primary", fail=True, delay=0.01)
        s = FakeLLM("secondary", fail=True, delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.03))

        with pytest.raises(RuntimeError):
            async for _ in hedged.stream("test prompt"):
                pass

    async def test_stream_with_messages(self) -> None:
        p = FakeLLM("primary", stream_chunks=["msg", " ", "tok"], delay=0.01)
        s = FakeLLM("secondary", delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.1))

        chunks: list[str] = []
        async for chunk in hedged.stream_with_messages([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert "".join(chunks) == "msg tok"
        assert hedged.last_winner == "primary"

    async def test_streaming_empty_chunks_handled(self) -> None:
        p = FakeLLM("primary", stream_chunks=[], delay=0.01)
        s = FakeLLM("secondary", stream_chunks=["backup"], delay=0.01)
        hedged = HedgedLLM(HedgingConfig(primary=p, secondary=s, hedge_delay=0.1))

        chunks: list[str] = []
        async for chunk in hedged.stream("test prompt"):
            chunks.append(chunk)

        assert chunks == []
        assert hedged.last_winner == "primary"
