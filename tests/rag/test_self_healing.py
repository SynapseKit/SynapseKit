from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.rag.self_healing import SelfHealingRAG


class _Strategy:
    def __init__(self, chunks):
        self.retrieve = AsyncMock(return_value=chunks)


@pytest.mark.asyncio
async def test_self_healing_single_strategy_success():
    llm = MagicMock()
    llm.tokens_used = {"input": 1, "output": 1}
    llm.generate_with_messages = AsyncMock(return_value="answer")

    metric = MagicMock()
    metric.evaluate = AsyncMock(return_value=MagicMock(score=0.9))

    rag = SelfHealingRAG(
        llm=llm,
        strategies=[_Strategy(["ctx"])],
        quality_threshold=0.75,
        max_retries=2,
        metric=metric,
    )

    answer = await rag.ask("q")
    assert answer == "answer"
    assert rag.last_report is not None
    assert rag.last_report.success is True
    assert rag.last_report.retries == 0


@pytest.mark.asyncio
async def test_self_healing_fallback_triggered():
    llm = MagicMock()
    llm.tokens_used = {"input": 1, "output": 1}
    llm.generate_with_messages = AsyncMock(side_effect=["bad", "good"])

    metric = MagicMock()
    metric.evaluate = AsyncMock(side_effect=[MagicMock(score=0.2), MagicMock(score=0.8)])

    primary = _Strategy(["ctx1"])
    fallback = _Strategy(["ctx2"])

    rag = SelfHealingRAG(
        llm=llm,
        strategies=[primary, fallback],
        quality_threshold=0.75,
        max_retries=2,
        metric=metric,
    )

    answer = await rag.ask("q")
    assert answer == "good"
    assert rag.last_report is not None
    assert rag.last_report.success is True
    assert rag.last_report.retries == 1


@pytest.mark.asyncio
async def test_self_healing_max_retries_cap():
    llm = MagicMock()
    llm.tokens_used = {"input": 1, "output": 1}
    llm.generate_with_messages = AsyncMock(return_value="ans")

    metric = MagicMock()
    metric.evaluate = AsyncMock(return_value=MagicMock(score=0.1))

    strategies = [_Strategy(["ctx1"]), _Strategy(["ctx2"]), _Strategy(["ctx3"])]

    rag = SelfHealingRAG(
        llm=llm,
        strategies=strategies,
        quality_threshold=0.9,
        max_retries=1,
        metric=metric,
    )

    answer = await rag.ask("q")
    assert answer == "ans"
    assert rag.last_report is not None
    # max_retries=1 => at most 2 attempts
    assert rag.last_report.attempts == 2
    assert rag.last_report.success is False


def test_empty_strategies_raises():
    with pytest.raises(ValueError, match="strategies must not be empty"):
        SelfHealingRAG(llm=MagicMock(), strategies=[])


def test_negative_max_retries_raises():
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        SelfHealingRAG(llm=MagicMock(), strategies=[_Strategy([])], max_retries=-1)


@pytest.mark.asyncio
async def test_strategy_exception_propagates_on_last_attempt():
    llm = MagicMock()
    llm.tokens_used = {"input": 1, "output": 1}
    llm.generate_with_messages = AsyncMock(return_value="ans")

    metric = MagicMock()
    metric.evaluate = AsyncMock(return_value=MagicMock(score=0.9))

    class _BrokenStrategy:
        async def retrieve(self, query, top_k=5, metadata_filter=None):
            raise RuntimeError("store unavailable")

    rag = SelfHealingRAG(
        llm=llm,
        strategies=[_BrokenStrategy()],
        quality_threshold=0.75,
        max_retries=0,
        metric=metric,
    )

    with pytest.raises(RuntimeError, match="store unavailable"):
        await rag.ask("q")

    assert rag.last_report is not None
    assert rag.last_report.success is False


def test_ask_sync_returns_str():
    llm = MagicMock()
    llm.tokens_used = {"input": 1, "output": 1}
    llm.generate_with_messages = AsyncMock(return_value="sync-answer")

    metric = MagicMock()
    metric.evaluate = AsyncMock(return_value=MagicMock(score=0.9))

    rag = SelfHealingRAG(
        llm=llm,
        strategies=[_Strategy(["ctx"])],
        quality_threshold=0.5,
        metric=metric,
    )

    result = rag.ask_sync("hello")
    assert result == "sync-answer"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_retrieve_falls_back_when_metadata_filter_not_accepted():
    """Retriever that doesn't accept metadata_filter should work via TypeError fallback."""
    llm = MagicMock()
    llm.tokens_used = {"input": 1, "output": 1}
    llm.generate_with_messages = AsyncMock(return_value="answer")

    metric = MagicMock()
    metric.evaluate = AsyncMock(return_value=MagicMock(score=0.9))

    class _NoFilterStrategy:
        async def retrieve(self, query, top_k=5):
            # No metadata_filter param — passing it causes TypeError
            return ["chunk without filter"]

    rag = SelfHealingRAG(
        llm=llm,
        strategies=[_NoFilterStrategy()],
        quality_threshold=0.5,
        metric=metric,
    )

    answer = await rag.ask("q", metadata_filter={"tag": "x"})
    assert answer == "answer"
