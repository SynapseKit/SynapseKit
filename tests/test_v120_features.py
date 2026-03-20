"""Tests for SynapseKit v1.2.0 features."""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# ───────────────────────────────────────────────────────────────────────
# CostTracker
# ───────────────────────────────────────────────────────────────────────


class TestCostTracker:
    def test_record_with_known_model(self):
        from synapsekit.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record("gpt-4o-mini", 1000, 500, 120.0)
        assert rec.model == "gpt-4o-mini"
        assert rec.input_tokens == 1000
        assert rec.output_tokens == 500
        assert rec.cost_usd > 0
        assert rec.latency_ms == 120.0

    def test_record_with_unknown_model(self):
        from synapsekit.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record("unknown-model", 100, 50, 10.0)
        assert rec.cost_usd == 0.0

    def test_scope_nesting(self):
        from synapsekit.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        with tracker.scope("pipeline"):
            with tracker.scope("retrieval"):
                tracker.record("gpt-4o-mini", 500, 200, 50.0)
            with tracker.scope("generation"):
                tracker.record("gpt-4o", 1000, 500, 200.0)

        summary = tracker.summary()
        assert "pipeline/retrieval" in summary
        assert "pipeline/generation" in summary
        assert summary["pipeline/retrieval"]["calls"] == 1
        assert summary["pipeline/generation"]["calls"] == 1

    def test_record_attribution_to_scope(self):
        from synapsekit.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 100, 50, 10.0)  # root scope
        with tracker.scope("inner"):
            tracker.record("gpt-4o", 200, 100, 20.0)

        summary = tracker.summary()
        assert "(root)" in summary
        assert "inner" in summary

    def test_total_cost_usd(self):
        from synapsekit.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 1000, 500, 100.0)
        tracker.record("gpt-4o-mini", 1000, 500, 100.0)
        assert tracker.total_cost_usd > 0
        # Two identical calls should double the cost
        single = tracker.records[0].cost_usd
        assert abs(tracker.total_cost_usd - 2 * single) < 1e-12

    def test_summary_aggregation(self):
        from synapsekit.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        with tracker.scope("s1"):
            tracker.record("gpt-4o-mini", 100, 50, 10.0)
            tracker.record("gpt-4o-mini", 200, 100, 20.0)

        summary = tracker.summary()
        assert summary["s1"]["calls"] == 2
        assert summary["s1"]["total_input_tokens"] == 300
        assert summary["s1"]["total_output_tokens"] == 150

    def test_cost_table_lookup(self):
        from synapsekit.observability.tracer import COST_TABLE

        assert "gpt-4o" in COST_TABLE
        assert "claude-sonnet-4-6" in COST_TABLE
        assert "input" in COST_TABLE["gpt-4o"]
        assert "output" in COST_TABLE["gpt-4o"]

    def test_reset(self):
        from synapsekit.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 100, 50, 10.0)
        tracker.reset()
        assert tracker.total_cost_usd == 0.0
        assert tracker.records == []
        assert tracker.summary() == {}


# ───────────────────────────────────────────────────────────────────────
# BudgetGuard
# ───────────────────────────────────────────────────────────────────────


class TestBudgetGuard:
    def test_per_request_limit(self):
        from synapsekit.observability.budget_guard import (
            BudgetExceededError,
            BudgetGuard,
            BudgetLimit,
        )

        guard = BudgetGuard(BudgetLimit(per_request=0.05))
        guard.check_before(0.04)  # OK
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.check_before(0.06)
        assert exc_info.value.limit_type == "per_request"

    def test_daily_limit(self):
        from synapsekit.observability.budget_guard import (
            BudgetExceededError,
            BudgetGuard,
            BudgetLimit,
        )

        guard = BudgetGuard(BudgetLimit(daily=0.10))
        guard.check_before(0.05)
        guard.record_spend(0.05)
        guard.check_before(0.04)
        guard.record_spend(0.04)
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.check_before(0.02)
        assert exc_info.value.limit_type == "daily"

    def test_per_user_limit(self):
        from synapsekit.observability.budget_guard import (
            BudgetExceededError,
            BudgetGuard,
            BudgetLimit,
        )

        guard = BudgetGuard(BudgetLimit(per_user=0.05))
        guard.check_before(0.03, user_id="alice")
        guard.record_spend(0.03, user_id="alice")
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.check_before(0.03, user_id="alice")
        assert exc_info.value.limit_type == "per_user"
        # Different user should be fine
        guard.check_before(0.03, user_id="bob")

    def test_circuit_breaker_states(self):
        from synapsekit.observability.budget_guard import (
            BudgetExceededError,
            BudgetGuard,
            BudgetLimit,
            CircuitState,
        )

        guard = BudgetGuard(BudgetLimit(daily=0.05), cooldown_seconds=0.1)
        assert guard.circuit_state == CircuitState.CLOSED

        guard.record_spend(0.04)
        # This should trip the circuit
        with pytest.raises(BudgetExceededError):
            guard.check_before(0.02)
        assert guard.circuit_state == CircuitState.OPEN

        # After cooldown, should transition to HALF_OPEN
        time.sleep(0.15)
        assert guard.circuit_state == CircuitState.HALF_OPEN

        # A small spend in HALF_OPEN should close the circuit
        guard.record_spend(0.001)
        # After record_spend in HALF_OPEN, if under daily, closes
        # but we already spent 0.041 > 0.05? No, 0.04 + 0.001 = 0.041 < 0.05
        assert guard.circuit_state == CircuitState.CLOSED

    def test_daily_reset(self):
        from synapsekit.observability.budget_guard import BudgetGuard, BudgetLimit

        guard = BudgetGuard(BudgetLimit(daily=0.10))
        guard.record_spend(0.05)
        assert guard.daily_spend == 0.05

        # Simulate day change
        guard._current_day = guard._today() - 1
        assert guard.daily_spend == 0.0

    def test_reset(self):
        from synapsekit.observability.budget_guard import (
            BudgetGuard,
            BudgetLimit,
            CircuitState,
        )

        guard = BudgetGuard(BudgetLimit(daily=0.10))
        guard.record_spend(0.05)
        guard.reset()
        assert guard.daily_spend == 0.0
        assert guard.circuit_state == CircuitState.CLOSED


# ───────────────────────────────────────────────────────────────────────
# @eval_case
# ───────────────────────────────────────────────────────────────────────


class TestEvalCase:
    def test_decorator_attaches_metadata(self):
        from synapsekit.evaluation.decorators import EvalCaseMeta, eval_case

        @eval_case(min_score=0.8, max_cost_usd=0.05, tags=["rag"])
        def my_eval():
            return {"score": 0.9}

        assert hasattr(my_eval, "_eval_case_meta")
        meta = my_eval._eval_case_meta
        assert isinstance(meta, EvalCaseMeta)
        assert meta.min_score == 0.8
        assert meta.max_cost_usd == 0.05
        assert meta.tags == ["rag"]

    def test_decorated_function_callable(self):
        from synapsekit.evaluation.decorators import eval_case

        @eval_case(min_score=0.5)
        def my_eval():
            return {"score": 0.7}

        result = my_eval()
        assert result == {"score": 0.7}

    def test_async_decorated_function(self):
        from synapsekit.evaluation.decorators import eval_case

        @eval_case(min_score=0.5)
        async def my_eval():
            return {"score": 0.7}

        result = asyncio.run(my_eval())
        assert result == {"score": 0.7}

    def test_default_metadata_values(self):
        from synapsekit.evaluation.decorators import eval_case

        @eval_case()
        def my_eval():
            return {"score": 1.0}

        meta = my_eval._eval_case_meta
        assert meta.min_score is None
        assert meta.max_cost_usd is None
        assert meta.max_latency_ms is None
        assert meta.tags == []

    def test_threshold_checking(self):
        """Test that _check_thresholds works correctly."""
        from synapsekit.cli.test import _check_thresholds
        from synapsekit.evaluation.decorators import EvalCaseMeta

        meta = EvalCaseMeta(min_score=0.8, max_cost_usd=0.05, max_latency_ms=1000)

        # Passing case
        passed, failures = _check_thresholds(
            {"score": 0.9, "cost_usd": 0.03, "latency_ms": 500}, meta, 0.7
        )
        assert passed
        assert failures == []

        # Failing score
        passed, failures = _check_thresholds(
            {"score": 0.5, "cost_usd": 0.03, "latency_ms": 500}, meta, 0.7
        )
        assert not passed
        assert any("score" in f for f in failures)

        # Failing cost
        passed, failures = _check_thresholds(
            {"score": 0.9, "cost_usd": 0.10, "latency_ms": 500}, meta, 0.7
        )
        assert not passed
        assert any("cost" in f for f in failures)

        # Failing latency
        passed, failures = _check_thresholds(
            {"score": 0.9, "cost_usd": 0.03, "latency_ms": 2000}, meta, 0.7
        )
        assert not passed
        assert any("latency" in f for f in failures)


# ───────────────────────────────────────────────────────────────────────
# PromptHub
# ───────────────────────────────────────────────────────────────────────


class TestPromptHub:
    def test_push_pull_roundtrip(self, tmp_path: Path):
        from synapsekit.prompts.hub import PromptHub

        hub = PromptHub(hub_dir=tmp_path)
        hub.push("my-org/summarize", "Summarize: {text}", version="v1")
        tpl = hub.pull("my-org/summarize:v1")
        assert tpl.format(text="hello") == "Summarize: hello"

    def test_latest_version_resolution(self, tmp_path: Path):
        from synapsekit.prompts.hub import PromptHub

        hub = PromptHub(hub_dir=tmp_path)
        hub.push("my-org/qa", "Q: {q}", version="v1")
        hub.push("my-org/qa", "Question: {q}", version="v2")
        # Pull without version → latest
        tpl = hub.pull("my-org/qa")
        assert tpl.format(q="test") == "Question: test"

    def test_list_prompts(self, tmp_path: Path):
        from synapsekit.prompts.hub import PromptHub

        hub = PromptHub(hub_dir=tmp_path)
        hub.push("org1/a", "A: {x}", version="v1")
        hub.push("org1/b", "B: {x}", version="v1")
        hub.push("org2/c", "C: {x}", version="v1")

        all_prompts = hub.list()
        assert set(all_prompts) == {"org1/a", "org1/b", "org2/c"}

        org1_prompts = hub.list(org="org1")
        assert set(org1_prompts) == {"org1/a", "org1/b"}

    def test_versions(self, tmp_path: Path):
        from synapsekit.prompts.hub import PromptHub

        hub = PromptHub(hub_dir=tmp_path)
        hub.push("my-org/qa", "Q: {q}", version="v1")
        hub.push("my-org/qa", "Question: {q}", version="v2")
        versions = hub.versions("my-org/qa")
        assert versions == ["v1", "v2"]

    def test_pull_nonexistent(self, tmp_path: Path):
        from synapsekit.prompts.hub import PromptHub

        hub = PromptHub(hub_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            hub.pull("no-org/no-prompt")

    def test_pull_nonexistent_version(self, tmp_path: Path):
        from synapsekit.prompts.hub import PromptHub

        hub = PromptHub(hub_dir=tmp_path)
        hub.push("my-org/qa", "Q: {q}", version="v1")
        with pytest.raises(FileNotFoundError):
            hub.pull("my-org/qa:v99")

    def test_invalid_ref(self, tmp_path: Path):
        from synapsekit.prompts.hub import PromptHub

        hub = PromptHub(hub_dir=tmp_path)
        with pytest.raises(ValueError):
            hub.pull("invalid-no-slash")


# ───────────────────────────────────────────────────────────────────────
# PluginRegistry
# ───────────────────────────────────────────────────────────────────────


class TestPluginRegistry:
    def test_discover_with_mock_entry_points(self):
        from synapsekit.plugins import PluginRegistry

        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = lambda: "registered"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry = PluginRegistry()
            names = registry.discover()
            assert "test_plugin" in names

    def test_load_plugin(self):
        from synapsekit.plugins import PluginRegistry

        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = lambda: {"name": "test"}

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry = PluginRegistry()
            result = registry.load("test_plugin")
            assert result == {"name": "test"}
            # Second load should return cached
            result2 = registry.load("test_plugin")
            assert result2 == {"name": "test"}

    def test_load_nonexistent(self):
        from synapsekit.plugins import PluginRegistry

        with patch("importlib.metadata.entry_points", return_value=[]):
            registry = PluginRegistry()
            with pytest.raises(KeyError):
                registry.load("nonexistent")

    def test_load_all(self):
        from synapsekit.plugins import PluginRegistry

        mock_ep1 = Mock()
        mock_ep1.name = "p1"
        mock_ep1.load.return_value = lambda: "r1"

        mock_ep2 = Mock()
        mock_ep2.name = "p2"
        mock_ep2.load.return_value = lambda: "r2"

        all_eps = [mock_ep1, mock_ep2]

        def ep_side_effect(**kwargs: Any) -> list[Mock]:
            name = kwargs.get("name")
            if name:
                return [ep for ep in all_eps if ep.name == name]
            return all_eps

        with patch("importlib.metadata.entry_points", side_effect=ep_side_effect):
            registry = PluginRegistry()
            all_loaded = registry.load_all()
            assert all_loaded == {"p1": "r1", "p2": "r2"}


# ───────────────────────────────────────────────────────────────────────
# RedisCheckpointer
# ───────────────────────────────────────────────────────────────────────


class TestRedisCheckpointer:
    def test_save_load_roundtrip(self):
        from synapsekit.graph.checkpointers.redis import RedisCheckpointer

        mock_client = MagicMock()
        store: dict[str, str] = {}

        def mock_set(key: str, value: str) -> None:
            store[key] = value

        def mock_get(key: str) -> str | None:
            return store.get(key)

        def mock_delete(key: str) -> None:
            store.pop(key, None)

        mock_client.set.side_effect = mock_set
        mock_client.get.side_effect = mock_get
        mock_client.delete.side_effect = mock_delete

        cp = RedisCheckpointer(mock_client)
        cp.save("graph-1", 5, {"messages": ["hello"]})
        result = cp.load("graph-1")
        assert result is not None
        step, state = result
        assert step == 5
        assert state == {"messages": ["hello"]}

    def test_load_nonexistent(self):
        from synapsekit.graph.checkpointers.redis import RedisCheckpointer

        mock_client = MagicMock()
        mock_client.get.return_value = None
        cp = RedisCheckpointer(mock_client)
        assert cp.load("nonexistent") is None

    def test_delete(self):
        from synapsekit.graph.checkpointers.redis import RedisCheckpointer

        mock_client = MagicMock()
        cp = RedisCheckpointer(mock_client)
        cp.delete("graph-1")
        mock_client.delete.assert_called_once()

    def test_ttl(self):
        from synapsekit.graph.checkpointers.redis import RedisCheckpointer

        mock_client = MagicMock()
        cp = RedisCheckpointer(mock_client, ttl=3600)
        cp.save("graph-1", 1, {"x": 1})
        mock_client.setex.assert_called_once()

    def test_close(self):
        from synapsekit.graph.checkpointers.redis import RedisCheckpointer

        mock_client = MagicMock()
        cp = RedisCheckpointer(mock_client)
        cp.close()
        mock_client.close.assert_called_once()


# ───────────────────────────────────────────────────────────────────────
# PostgresCheckpointer
# ───────────────────────────────────────────────────────────────────────


class TestPostgresCheckpointer:
    def _make_mock_conn(self) -> tuple[MagicMock, dict[str, Any]]:
        """Create a mock psycopg connection with an in-memory store."""
        store: dict[str, Any] = {}
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        def mock_execute(sql: str, params: tuple | None = None) -> None:
            if "CREATE TABLE" in sql:
                return
            if "INSERT" in sql and params:
                graph_id, step, state_json = params
                store[graph_id] = (step, state_json)
            elif "SELECT" in sql and params:
                graph_id = params[0]
                if graph_id in store:
                    step, state_json = store[graph_id]
                    mock_cursor.fetchone.return_value = (step, state_json)
                else:
                    mock_cursor.fetchone.return_value = None
            elif "DELETE" in sql and params:
                store.pop(params[0], None)

        mock_cursor.execute.side_effect = mock_execute
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        return mock_conn, store

    def test_save_load_roundtrip(self):
        from synapsekit.graph.checkpointers.postgres import PostgresCheckpointer

        mock_conn, _ = self._make_mock_conn()
        cp = PostgresCheckpointer(mock_conn)
        cp.save("graph-1", 3, {"messages": ["hello"]})
        result = cp.load("graph-1")
        assert result is not None
        step, _state = result
        assert step == 3

    def test_load_nonexistent(self):
        from synapsekit.graph.checkpointers.postgres import PostgresCheckpointer

        mock_conn, _ = self._make_mock_conn()
        cp = PostgresCheckpointer(mock_conn)
        assert cp.load("nonexistent") is None

    def test_delete(self):
        from synapsekit.graph.checkpointers.postgres import PostgresCheckpointer

        mock_conn, _store = self._make_mock_conn()
        cp = PostgresCheckpointer(mock_conn)
        cp.save("graph-1", 1, {"x": 1})
        cp.delete("graph-1")
        assert cp.load("graph-1") is None

    def test_close(self):
        from synapsekit.graph.checkpointers.postgres import PostgresCheckpointer

        mock_conn, _ = self._make_mock_conn()
        cp = PostgresCheckpointer(mock_conn)
        cp.close()
        mock_conn.close.assert_called_once()

    def test_upsert(self):
        from synapsekit.graph.checkpointers.postgres import PostgresCheckpointer

        mock_conn, _ = self._make_mock_conn()
        cp = PostgresCheckpointer(mock_conn)
        cp.save("graph-1", 1, {"step": "first"})
        cp.save("graph-1", 2, {"step": "second"})
        result = cp.load("graph-1")
        assert result is not None
        assert result[0] == 2


# ───────────────────────────────────────────────────────────────────────
# CLI serve
# ───────────────────────────────────────────────────────────────────────


class TestCLIServe:
    def test_detect_type_rag(self):
        from synapsekit.cli.serve import _detect_type

        class RAGPipeline:
            pass

        assert _detect_type(RAGPipeline()) == "rag"

    def test_detect_type_graph(self):
        from synapsekit.cli.serve import _detect_type

        class CompiledGraph:
            pass

        assert _detect_type(CompiledGraph()) == "graph"

    def test_detect_type_agent(self):
        from synapsekit.cli.serve import _detect_type

        class ReActAgent:
            pass

        assert _detect_type(ReActAgent()) == "agent"

    def test_detect_type_fallback(self):
        from synapsekit.cli.serve import _detect_type

        class SomethingElse:
            pass

        assert _detect_type(SomethingElse()) == "agent"

    def test_build_app_rag(self):
        pytest.importorskip("fastapi")
        from synapsekit.cli.serve import build_app

        mock_rag = MagicMock()
        app = build_app(mock_rag, app_type="rag")
        routes = [r.path for r in app.routes]
        assert "/query" in routes
        assert "/health" in routes

    def test_build_app_graph(self):
        pytest.importorskip("fastapi")
        from synapsekit.cli.serve import build_app

        mock_graph = MagicMock()
        app = build_app(mock_graph, app_type="graph")
        routes = [r.path for r in app.routes]
        assert "/run" in routes
        assert "/stream" in routes
        assert "/health" in routes

    def test_build_app_agent(self):
        pytest.importorskip("fastapi")
        from synapsekit.cli.serve import build_app

        mock_agent = MagicMock()
        app = build_app(mock_agent, app_type="agent")
        routes = [r.path for r in app.routes]
        assert "/run" in routes
        assert "/health" in routes

    def test_health_endpoint(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from starlette.testclient import TestClient

        from synapsekit.cli.serve import build_app

        mock_rag = MagicMock()
        app = build_app(mock_rag, app_type="rag")
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_rag_query_endpoint(self):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from starlette.testclient import TestClient

        from synapsekit.cli.serve import build_app

        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="The answer is 42")

        app = build_app(mock_rag, app_type="rag")
        client = TestClient(app)
        resp = client.post("/query", json={"query": "What is the answer?"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "The answer is 42"

    def test_import_object_invalid(self):
        from synapsekit.cli.serve import _import_object

        with pytest.raises(ValueError, match="module:attribute"):
            _import_object("no_colon_here")


# ───────────────────────────────────────────────────────────────────────
# CLI test
# ───────────────────────────────────────────────────────────────────────


class TestCLITest:
    def test_discover_eval_files(self, tmp_path: Path):
        from synapsekit.cli.test import _discover_eval_files

        (tmp_path / "eval_qa.py").touch()
        (tmp_path / "regression_eval.py").touch()
        (tmp_path / "not_a_test.py").touch()

        files = _discover_eval_files(str(tmp_path))
        filenames = {f.name for f in files}
        assert "eval_qa.py" in filenames
        assert "regression_eval.py" in filenames
        assert "not_a_test.py" not in filenames

    def test_check_thresholds_pass(self):
        from synapsekit.cli.test import _check_thresholds
        from synapsekit.evaluation.decorators import EvalCaseMeta

        meta = EvalCaseMeta(min_score=0.7)
        passed, failures = _check_thresholds({"score": 0.8}, meta, 0.7)
        assert passed
        assert failures == []

    def test_check_thresholds_fail(self):
        from synapsekit.cli.test import _check_thresholds
        from synapsekit.evaluation.decorators import EvalCaseMeta

        meta = EvalCaseMeta(min_score=0.9)
        passed, _failures = _check_thresholds({"score": 0.5}, meta, 0.7)
        assert not passed

    def test_check_thresholds_uses_global(self):
        from synapsekit.cli.test import _check_thresholds
        from synapsekit.evaluation.decorators import EvalCaseMeta

        meta = EvalCaseMeta()  # No min_score set
        passed, _failures = _check_thresholds({"score": 0.5}, meta, 0.7)
        assert not passed  # Uses global threshold 0.7

    def test_find_eval_cases(self):
        from synapsekit.cli.test import _find_eval_cases
        from synapsekit.evaluation.decorators import eval_case

        mod = type(sys)("test_mod")

        @eval_case(min_score=0.8)
        def eval_1():
            return {"score": 0.9}

        def not_eval():
            pass

        mod.eval_1 = eval_1
        mod.not_eval = not_eval

        cases = _find_eval_cases(mod)
        names = [n for n, _ in cases]
        assert "eval_1" in names
        assert "not_eval" not in names

    def test_run_test_exit_code(self, tmp_path: Path):
        from synapsekit.cli.test import run_test

        eval_file = tmp_path / "eval_test.py"
        eval_file.write_text(
            textwrap.dedent("""
            from synapsekit.evaluation.decorators import eval_case

            @eval_case(min_score=0.9)
            def eval_failing():
                return {"score": 0.3}
        """)
        )

        args = Mock()
        args.path = str(tmp_path)
        args.threshold = 0.7
        args.output_format = "table"

        with pytest.raises(SystemExit) as exc_info:
            run_test(args)
        assert exc_info.value.code == 1

    def test_run_test_passing(self, tmp_path: Path):
        from synapsekit.cli.test import run_test

        eval_file = tmp_path / "eval_pass.py"
        eval_file.write_text(
            textwrap.dedent("""
            from synapsekit.evaluation.decorators import eval_case

            @eval_case(min_score=0.5)
            def eval_passing():
                return {"score": 0.9}
        """)
        )

        args = Mock()
        args.path = str(tmp_path)
        args.threshold = 0.7
        args.output_format = "json"

        # Should not raise SystemExit
        run_test(args)

    def test_run_test_json_format(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        from synapsekit.cli.test import run_test

        eval_file = tmp_path / "eval_json.py"
        eval_file.write_text(
            textwrap.dedent("""
            from synapsekit.evaluation.decorators import eval_case

            @eval_case(min_score=0.5)
            def eval_good():
                return {"score": 0.9, "cost_usd": 0.01}
        """)
        )

        args = Mock()
        args.path = str(tmp_path)
        args.threshold = 0.7
        args.output_format = "json"

        run_test(args)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["passed"] is True


# ───────────────────────────────────────────────────────────────────────
# CLI main
# ───────────────────────────────────────────────────────────────────────


class TestCLIMain:
    def test_version_flag(self, capsys: pytest.CaptureFixture[str]):
        from synapsekit.cli.main import main

        main(["--version"])
        output = capsys.readouterr().out
        assert "1.2.0" in output

    def test_no_command(self):
        from synapsekit.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1


# ───────────────────────────────────────────────────────────────────────
# Top-level imports
# ───────────────────────────────────────────────────────────────────────


class TestTopLevelImports:
    def test_cost_tracker_import(self):
        from synapsekit import CostTracker

        assert CostTracker is not None

    def test_budget_guard_import(self):
        from synapsekit import BudgetExceededError, BudgetGuard, BudgetLimit, CircuitState

        assert BudgetGuard is not None
        assert BudgetLimit is not None
        assert BudgetExceededError is not None
        assert CircuitState is not None

    def test_eval_case_import(self):
        from synapsekit import EvalCaseMeta, eval_case

        assert eval_case is not None
        assert EvalCaseMeta is not None

    def test_prompt_hub_import(self):
        from synapsekit import PromptHub

        assert PromptHub is not None

    def test_plugin_registry_import(self):
        from synapsekit import PluginRegistry

        assert PluginRegistry is not None

    def test_redis_checkpointer_import(self):
        from synapsekit import RedisCheckpointer

        assert RedisCheckpointer is not None

    def test_postgres_checkpointer_import(self):
        from synapsekit import PostgresCheckpointer

        assert PostgresCheckpointer is not None

    def test_version_bumped(self):
        import synapsekit

        assert synapsekit.__version__ == "1.2.0"
