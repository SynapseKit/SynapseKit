import sys
import types

import pytest

from synapsekit.evaluation.benchmarks.agentbench import AgentBenchBenchmark
from synapsekit.evaluation.benchmarks.base import BenchmarkResult
from synapsekit.evaluation.benchmarks.gaia import GAIABenchmark
from synapsekit.evaluation.benchmarks.humaneval import HumanEvalBenchmark
from synapsekit.evaluation.benchmarks.swe_bench import SWEBenchmark
from synapsekit.evaluation.benchmarks.webarena import WebArenaBenchmark


@pytest.fixture
def fake_datasets(monkeypatch):
    """Install a minimal fake `datasets` module in sys.modules.

    Benchmarks do `from datasets import load_dataset` lazily inside
    `load_dataset()`, so tests don't need the real (heavy, network-calling)
    `datasets` package installed to exercise the loading/evaluation logic.
    """
    records = [{"input": "task-1", "expected_answer": "42"}, {"input": "task-2"}]

    def load_dataset(*args, **kwargs):
        return list(records)

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    return records


def test_benchmark_names():
    assert GAIABenchmark().name == "GAIA"
    assert SWEBenchmark().name == "SWE-bench"
    assert WebArenaBenchmark().name == "WebArena"
    assert AgentBenchBenchmark().name == "AgentBench"
    assert HumanEvalBenchmark().name == "HumanEval"


def test_gaia_benchmark_evaluate_matches_expected_answer(fake_datasets):
    def mock_agent(task):
        return task["input"]

    benchmark = GAIABenchmark()
    result = benchmark.evaluate(mock_agent)

    assert isinstance(result, BenchmarkResult)
    assert result.benchmark_name == "GAIA"
    assert result.total_tasks == 2
    # task-1 has expected_answer "42" but the agent echoes "task-1" -> no match.
    # task-2 has no expected_answer -> placeholder (non-null) grading applies.
    assert result.successful_tasks == 1
    assert result.score == 0.5

    leaderboard = result.format_leaderboard()
    assert "GAIA Leaderboard" in leaderboard
    assert "Score: 0.5000" in leaderboard


@pytest.mark.parametrize(
    "benchmark_cls",
    [AgentBenchBenchmark, SWEBenchmark, WebArenaBenchmark, HumanEvalBenchmark],
)
def test_placeholder_benchmarks_count_non_null_responses(fake_datasets, benchmark_cls):
    def mock_agent(task):
        return task["input"]

    benchmark = benchmark_cls()
    result = benchmark.evaluate(mock_agent)

    assert result.total_tasks == 2
    assert result.successful_tasks == 2
    assert result.score == 1.0


def test_benchmark_evaluate_records_agent_exceptions(fake_datasets):
    def failing_agent(task):
        raise RuntimeError("boom")

    benchmark = GAIABenchmark()
    result = benchmark.evaluate(failing_agent)

    assert result.total_tasks == 2
    assert result.successful_tasks == 0
    assert result.details["errors"] == ["boom", "boom"]


def test_load_dataset_without_datasets_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(ImportError, match="synapsekit\\[huggingface\\]"):
        GAIABenchmark().load_dataset()
