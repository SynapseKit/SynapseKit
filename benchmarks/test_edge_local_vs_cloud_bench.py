"""CI-enforced check that the edge benchmark harness itself is correct.

CI has no GGUF file and no ``ANTHROPIC_API_KEY``, so this cannot assert on real
local-vs-cloud quality -- those numbers come from running
``edge_local_vs_cloud_bench.py`` by hand and land in ``docs/edge.md``. What it
*can* pin down, and what actually rots silently, is the harness: that graders
score known-good and known-bad answers correctly, that every task produces a
record, and that latency and cost are measured per model.

A grader that quietly starts returning 1.0 for everything would make the
published benchmark meaningless, which is why the grader cases below are
exhaustive over the four grader modes rather than a smoke test.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from edge_local_vs_cloud_bench import accuracy_by_category, run_suite, summarize
from edge_task_suite import CATEGORIES, EdgeTask, generate_tasks, grade

from synapsekit.llm.base import BaseLLM, LLMConfig

_N_TASKS = 20
_SEED = 7


class StubLLM(BaseLLM):
    """Answers each task correctly or incorrectly on demand, with no network."""

    def __init__(self, model: str, *, correct: bool) -> None:
        super().__init__(LLMConfig(model=model, api_key="", provider="stub"))
        self._correct = correct
        self.prompts: list[str] = []

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        self.prompts.append(prompt)
        self._input_tokens += 10
        self._output_tokens += 5
        yield _ideal_answer(prompt) if self._correct else "definitely wrong"


def _ideal_answer(prompt: str) -> str:
    """Produce a response that should score 1.0 for the task with this prompt."""
    task = _TASKS_BY_PROMPT[prompt]
    if task.category == "json":
        fields = dict(pair.split("=", 1) for pair in task.expected.split(";"))
        return f'{{"name": "{fields["name"]}", "age": {fields["age"]}}}'
    if task.category == "format":
        return task.expected.replace(r"\b", "")
    return task.expected


_TASKS = generate_tasks(n=_N_TASKS, seed=_SEED)
_TASKS_BY_PROMPT = {t.prompt: t for t in _TASKS}


# --------------------------------------------------------------------------- #
# Graders
# --------------------------------------------------------------------------- #


def test_every_category_is_represented() -> None:
    assert {t.category for t in _TASKS} == set(CATEGORIES)


def test_generate_tasks_is_deterministic_for_a_seed() -> None:
    assert generate_tasks(n=_N_TASKS, seed=_SEED) == _TASKS


def test_graders_accept_the_ideal_answer() -> None:
    for task in _TASKS:
        assert grade(task, _ideal_answer(task.prompt)) == 1.0, task.name


def test_graders_reject_a_wrong_answer() -> None:
    for task in _TASKS:
        assert grade(task, "definitely wrong") == 0.0, task.name


def test_graders_tolerate_surrounding_prose() -> None:
    for task in _TASKS:
        padded = f"Sure! The answer is {_ideal_answer(task.prompt)} -- hope that helps."
        assert grade(task, padded) == 1.0, task.name


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("negative", 0.0),
        ("this is not positive at all", 0.0),
        ("positive", 1.0),
    ],
)
def test_label_grader_uses_first_label_mentioned(output: str, expected: float) -> None:
    task = EdgeTask("t", "classification", "p", "positive", "label")
    assert grade(task, output) == expected


def test_json_grader_rejects_wrong_field_value() -> None:
    task = EdgeTask("t", "json", "p", "name=Ivan Kim;age=41", "json_field")
    assert grade(task, '{"name": "Ivan Kim", "age": 41}') == 1.0
    assert grade(task, '{"name": "Ivan Kim", "age": 42}') == 0.0
    assert grade(task, "no json here") == 0.0


def test_json_grader_finds_json_inside_a_fenced_block() -> None:
    task = EdgeTask("t", "json", "p", "name=Hana Voss;age=30", "json_field")
    fenced = 'Here you go:\n```json\n{"name": "Hana Voss", "age": 30}\n```'
    assert grade(task, fenced) == 1.0


def test_unknown_grader_raises() -> None:
    with pytest.raises(ValueError, match="unknown grader"):
        grade(EdgeTask("t", "extraction", "p", "x", "nope"), "x")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def suite_records() -> tuple[list, list]:
    async def _run() -> tuple[list, list]:
        good = await run_suite(StubLLM("local-stub", correct=True), _TASKS)
        bad = await run_suite(StubLLM("claude-haiku-4-5-20251001", correct=False), _TASKS)
        return good, bad

    return asyncio.run(_run())


def test_every_task_produces_one_record(suite_records) -> None:
    good, bad = suite_records
    assert len(good) == len(bad) == _N_TASKS
    assert [r.name for r in good] == [t.name for t in _TASKS]


def test_summary_separates_a_correct_model_from_a_wrong_one(suite_records) -> None:
    good, bad = suite_records
    assert summarize(good)["accuracy"] == 1.0
    assert summarize(bad)["accuracy"] == 0.0


def test_latency_is_recorded_for_every_task(suite_records) -> None:
    good, _ = suite_records
    assert all(r.latency_ms is not None and r.latency_ms >= 0 for r in good)


def test_cost_is_zero_for_unpriced_local_models_and_nonzero_for_cloud(
    suite_records,
) -> None:
    good, bad = suite_records
    assert summarize(good)["total_cost_usd"] == 0.0
    assert summarize(bad)["total_cost_usd"] > 0.0


def test_accuracy_by_category_covers_every_category(suite_records) -> None:
    good, _ = suite_records
    by_category = accuracy_by_category(good)
    assert set(by_category) == set(CATEGORIES)
    assert all(score == 1.0 for score in by_category.values())


def test_provider_failure_scores_zero_instead_of_crashing() -> None:
    class BoomLLM(BaseLLM):
        def __init__(self) -> None:
            super().__init__(LLMConfig(model="boom", api_key="", provider="stub"))

        async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
            raise RuntimeError("model exploded")
            yield ""  # pragma: no cover - unreachable, makes this an async generator

    records = asyncio.run(run_suite(BoomLLM(), _TASKS[:3]))
    assert len(records) == 3
    assert all(r.score == 0.0 for r in records)
    assert all("model exploded" in (r.output or "") for r in records)
