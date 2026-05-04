from unittest.mock import AsyncMock, MagicMock

import pytest

from synapsekit.rag.self_healing import SelfHealingRAG


class _Strategy:
    def __init__(self, chunks):
        self.retrieve = AsyncMock(return_value=chunks)


@pytest.mark.asyncio
async def test_self_healing_real_metric_integration():
    llm = MagicMock()
    # Mock for _generate_answer
    llm.generate_with_messages = AsyncMock(return_value="The capital of France is Paris.")

    # Mock for FaithfulnessMetric.evaluate
    # It calls llm.generate twice: once for claims, once for support check
    llm.generate = AsyncMock(
        side_effect=[
            "1. The capital of France is Paris.",  # claims response
            "YES",  # check response
        ]
    )

    strategy = _Strategy(["Paris is the capital of France."])

    rag = SelfHealingRAG(
        llm=llm,
        strategies=[strategy],
        quality_threshold=0.8,
    )

    answer = await rag.ask("What is the capital of France?")
    assert answer == "The capital of France is Paris."
    assert rag.last_report is not None
    assert rag.last_report.success is True
    assert rag.last_report.scores[0] == 1.0
