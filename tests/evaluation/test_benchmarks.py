import pytest

from synapsekit.evaluation.benchmarks.agentbench import AgentBenchBenchmark
from synapsekit.evaluation.benchmarks.base import BenchmarkResult
from synapsekit.evaluation.benchmarks.gaia import GAIABenchmark
from synapsekit.evaluation.benchmarks.swe_bench import SWEBenchmark
from synapsekit.evaluation.benchmarks.webarena import WebArenaBenchmark


def test_benchmark_names():
    assert GAIABenchmark().name == "GAIA"
    assert SWEBenchmark().name == "SWE-bench"
    assert WebArenaBenchmark().name == "WebArena"
    assert AgentBenchBenchmark().name == "AgentBench"


def test_benchmark_evaluate():
    def mock_agent(task):
        return task["input"]

    benchmark = GAIABenchmark()
    # Stub returns [] currently
    result = benchmark.evaluate(mock_agent)
    assert isinstance(result, BenchmarkResult)
    assert result.benchmark_name == "GAIA"
    assert result.total_tasks == 0
    assert result.successful_tasks == 0
    assert result.score == 0.0

    # Ensure leaderboard format doesn't crash
    leaderboard = result.format_leaderboard()
    assert "GAIA Leaderboard" in leaderboard
    assert "Score: 0.0000" in leaderboard
