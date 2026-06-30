from __future__ import annotations

import asyncio

import pytest

from synapsekit.evaluation import EvalSuite, eval_case


@eval_case(min_score=0.8)
async def _prompt_case(prompt: str = ""):
    return {"score": 0.9 if "better" in prompt else 0.7, "cost_usd": 0.01}


def test_eval_suite_scores_prompt_cases():
    suite = EvalSuite.from_cases([("prompt_case", _prompt_case)], threshold=0.8)

    result = asyncio.run(suite.score_prompt("better prompt"))

    assert result.passed is True
    assert result.score == pytest.approx(0.9)
    assert result.cost_usd == pytest.approx(0.01)
    assert result.n_cases == 1


def test_eval_suite_fails_threshold():
    suite = EvalSuite.from_cases([("prompt_case", _prompt_case)], threshold=0.8)

    result = asyncio.run(suite.score_prompt("base prompt"))

    assert result.passed is False
    assert result.failures == ["score 0.700 < min 0.800"]


def test_eval_suite_empty_suite_is_blocking():
    suite = EvalSuite.from_cases([], threshold=0.1)

    result = asyncio.run(suite.score_prompt("anything"))

    assert result.passed is False
    assert result.score == 0.0
    assert result.failures == ["no eval cases found"]
