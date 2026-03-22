"""Tests for EvalRegression and EvalCI (v1.3.0)."""

from __future__ import annotations

import json

import pytest

from synapsekit.evaluation.regression import (
    DEFAULT_THRESHOLDS,
    EvalRegression,
    EvalSnapshot,
    MetricDelta,
    RegressionReport,
)


@pytest.fixture
def store_dir(tmp_path):
    return str(tmp_path / "evals")


@pytest.fixture
def sample_results():
    return [
        {"name": "case_a", "score": 0.9, "cost_usd": 0.01, "latency_ms": 100},
        {"name": "case_b", "score": 0.8, "cost_usd": 0.02, "latency_ms": 200},
    ]


class TestEvalRegression:
    def test_save_snapshot(self, store_dir, sample_results):
        reg = EvalRegression(store_dir=store_dir)
        path = reg.save_snapshot("baseline", sample_results)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "baseline"
        assert len(data["results"]) == 2

    def test_load_snapshot(self, store_dir, sample_results):
        reg = EvalRegression(store_dir=store_dir)
        reg.save_snapshot("v1", sample_results)
        snap = reg.load_snapshot("v1")
        assert isinstance(snap, EvalSnapshot)
        assert snap.name == "v1"
        assert len(snap.results) == 2

    def test_load_missing_raises(self, store_dir):
        reg = EvalRegression(store_dir=store_dir)
        with pytest.raises(FileNotFoundError):
            reg.load_snapshot("nonexistent")

    def test_list_snapshots(self, store_dir, sample_results):
        reg = EvalRegression(store_dir=store_dir)
        reg.save_snapshot("alpha", sample_results)
        reg.save_snapshot("beta", sample_results)
        names = reg.list_snapshots()
        assert names == ["alpha", "beta"]

    def test_compare_no_regression(self, store_dir, sample_results):
        reg = EvalRegression(store_dir=store_dir)
        reg.save_snapshot("v1", sample_results)
        reg.save_snapshot("v2", sample_results)  # identical
        report = reg.compare("v1", "v2")
        assert isinstance(report, RegressionReport)
        assert not report.has_regressions

    def test_compare_score_regression(self, store_dir, sample_results):
        reg = EvalRegression(store_dir=store_dir)
        reg.save_snapshot("v1", sample_results)
        worse = [
            {"name": "case_a", "score": 0.85, "cost_usd": 0.01, "latency_ms": 100},
            {"name": "case_b", "score": 0.8, "cost_usd": 0.02, "latency_ms": 200},
        ]
        reg.save_snapshot("v2", worse)
        report = reg.compare("v1", "v2")
        assert report.has_regressions
        score_deltas = [d for d in report.deltas if d.metric == "score" and d.regressed]
        assert len(score_deltas) == 1
        assert score_deltas[0].case_name == "case_a"

    def test_compare_cost_regression(self, store_dir, sample_results):
        reg = EvalRegression(store_dir=store_dir)
        reg.save_snapshot("v1", sample_results)
        expensive = [
            {"name": "case_a", "score": 0.9, "cost_usd": 0.05, "latency_ms": 100},
            {"name": "case_b", "score": 0.8, "cost_usd": 0.02, "latency_ms": 200},
        ]
        reg.save_snapshot("v2", expensive)
        report = reg.compare("v1", "v2")
        assert report.has_regressions
        cost_deltas = [d for d in report.deltas if d.metric == "cost_usd" and d.regressed]
        assert len(cost_deltas) >= 1

    def test_compare_latency_regression(self, store_dir, sample_results):
        reg = EvalRegression(store_dir=store_dir)
        reg.save_snapshot("v1", sample_results)
        slow = [
            {"name": "case_a", "score": 0.9, "cost_usd": 0.01, "latency_ms": 150},
            {"name": "case_b", "score": 0.8, "cost_usd": 0.02, "latency_ms": 200},
        ]
        reg.save_snapshot("v2", slow)
        report = reg.compare("v1", "v2")
        latency_deltas = [d for d in report.deltas if d.metric == "latency_ms" and d.regressed]
        assert len(latency_deltas) >= 1

    def test_custom_thresholds(self, store_dir, sample_results):
        reg = EvalRegression(store_dir=store_dir)
        reg.save_snapshot("v1", sample_results)
        slightly_worse = [
            {"name": "case_a", "score": 0.895, "cost_usd": 0.01, "latency_ms": 100},
            {"name": "case_b", "score": 0.8, "cost_usd": 0.02, "latency_ms": 200},
        ]
        reg.save_snapshot("v2", slightly_worse)
        # With default thresholds (-0.02), this should NOT regress
        report = reg.compare("v1", "v2")
        score_regressed = [d for d in report.deltas if d.metric == "score" and d.regressed]
        assert len(score_regressed) == 0

        # With tight threshold, it SHOULD regress
        report2 = reg.compare("v1", "v2", thresholds={"score": -0.001})
        score_regressed2 = [d for d in report2.deltas if d.metric == "score" and d.regressed]
        assert len(score_regressed2) == 1

    def test_default_thresholds(self):
        assert DEFAULT_THRESHOLDS["score"] == -0.02
        assert DEFAULT_THRESHOLDS["cost_usd"] == 0.10
        assert DEFAULT_THRESHOLDS["latency_ms"] == 0.20

    def test_metric_delta_dataclass(self):
        d = MetricDelta("case", "score", 0.9, 0.85, -0.05, True)
        assert d.regressed
        assert d.delta == -0.05

    def test_regression_report_no_deltas(self):
        report = RegressionReport(baseline_name="a", current_name="b")
        assert not report.has_regressions


class TestEvalCI:
    """Test CLI-level regression flags via the test subparser."""

    def test_cli_has_save_flag(self):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        from synapsekit.cli.main import _add_test_parser

        _add_test_parser(subparsers)
        args = parser.parse_args(["test", "--save", "v1"])
        assert args.save_snapshot == "v1"

    def test_cli_has_compare_flag(self):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        from synapsekit.cli.main import _add_test_parser

        _add_test_parser(subparsers)
        args = parser.parse_args(["test", "--compare", "baseline"])
        assert args.compare_baseline == "baseline"

    def test_cli_has_fail_on_regression_flag(self):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        from synapsekit.cli.main import _add_test_parser

        _add_test_parser(subparsers)
        args = parser.parse_args(["test", "--fail-on-regression"])
        assert args.fail_on_regression is True
