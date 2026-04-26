import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from synapsekit.cli.benchmark import run_benchmark


def test_run_benchmark_list(capsys):
    args = argparse.Namespace(benchmark_command="list")
    run_benchmark(args)
    
    captured = capsys.readouterr()
    assert "Available Benchmarks:" in captured.out
    assert "gaia (GAIA)" in captured.out
    assert "swe-bench (SWE-bench)" in captured.out


def test_run_benchmark_run_unknown_suite():
    args = argparse.Namespace(benchmark_command="run", suite="unknown", agent="my_agent:run", split="test", limit=None)
    with pytest.raises(SystemExit) as exc_info:
        run_benchmark(args)
    assert "Unknown benchmark suite: unknown" in str(exc_info.value)


@patch("importlib.import_module")
def test_run_benchmark_run_success(mock_import_module, capsys):
    mock_module = MagicMock()
    def mock_agent(task): return "success"
    mock_module.mock_agent = mock_agent
    mock_import_module.return_value = mock_module

    args = argparse.Namespace(benchmark_command="run", suite="gaia", agent="my_module:mock_agent", split="test", limit=None)
    run_benchmark(args)

    captured = capsys.readouterr()
    assert "Running benchmark GAIA..." in captured.out
    assert "GAIA Leaderboard" in captured.out


def test_run_benchmark_missing_subcommand():
    args = argparse.Namespace(benchmark_command=None)
    with pytest.raises(SystemExit) as exc_info:
        run_benchmark(args)
    assert "Missing benchmark subcommand. Use: run or list" in str(exc_info.value)
