"""Integration tests for v1.2.0 serve endpoints and cross-module behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from synapsekit.cli.serve import build_app  # noqa: E402

# ───────────────────────────────────────────────────────────────────────
# RAG endpoint integration
# ───────────────────────────────────────────────────────────────────────


class TestRAGEndpoints:
    def _make_rag(self, answer: str = "42") -> MagicMock:
        rag = MagicMock()
        rag.__class__.__name__ = "RAGPipeline"

        async def aquery(q: str) -> str:
            return answer

        rag.aquery = aquery
        return rag

    def test_health(self):
        app = build_app(self._make_rag(), app_type="rag")
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_query_success(self):
        app = build_app(self._make_rag("The answer is 42"), app_type="rag")
        client = TestClient(app)
        r = client.post("/query", json={"query": "What is the meaning of life?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "The answer is 42"

    def test_query_with_question_key(self):
        app = build_app(self._make_rag("yes"), app_type="rag")
        client = TestClient(app)
        r = client.post("/query", json={"question": "Is this working?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "yes"

    def test_query_missing_field(self):
        app = build_app(self._make_rag(), app_type="rag")
        client = TestClient(app)
        r = client.post("/query", json={"foo": "bar"})
        assert r.status_code == 400
        assert "error" in r.json()

    def test_openapi_docs(self):
        app = build_app(self._make_rag(), app_type="rag")
        client = TestClient(app)
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "/query" in schema["paths"]
        assert "/health" in schema["paths"]


# ───────────────────────────────────────────────────────────────────────
# Graph endpoint integration
# ───────────────────────────────────────────────────────────────────────


class TestGraphEndpoints:
    def _make_graph(self, result: dict[str, Any] | None = None) -> MagicMock:
        graph = MagicMock()
        graph.__class__.__name__ = "CompiledGraph"

        async def arun(state: dict) -> dict:
            return result or {"output": "done", **state}

        graph.arun = arun
        return graph

    def test_run_success(self):
        app = build_app(self._make_graph({"output": "completed"}), app_type="graph")
        client = TestClient(app)
        r = client.post("/run", json={"state": {"messages": ["hello"]}})
        assert r.status_code == 200
        assert r.json()["result"]["output"] == "completed"

    def test_stream_endpoint_exists(self):
        app = build_app(self._make_graph(), app_type="graph")
        client = TestClient(app)
        r = client.get("/stream?state=%7B%7D")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

    def test_health(self):
        app = build_app(self._make_graph(), app_type="graph")
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200

    def test_openapi_docs(self):
        app = build_app(self._make_graph(), app_type="graph")
        client = TestClient(app)
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "/run" in schema["paths"]
        assert "/stream" in schema["paths"]


# ───────────────────────────────────────────────────────────────────────
# Agent endpoint integration
# ───────────────────────────────────────────────────────────────────────


class TestAgentEndpoints:
    def _make_agent(self, answer: str = "I can help") -> MagicMock:
        agent = MagicMock()
        agent.__class__.__name__ = "ReActAgent"

        async def arun(prompt: str) -> str:
            return answer

        agent.arun = arun
        return agent

    def test_run_success(self):
        app = build_app(self._make_agent("Hello!"), app_type="agent")
        client = TestClient(app)
        r = client.post("/run", json={"prompt": "Hi there"})
        assert r.status_code == 200
        assert r.json()["answer"] == "Hello!"

    def test_run_with_input_key(self):
        app = build_app(self._make_agent("Sure"), app_type="agent")
        client = TestClient(app)
        r = client.post("/run", json={"input": "Do something"})
        assert r.status_code == 200
        assert r.json()["answer"] == "Sure"

    def test_run_missing_prompt(self):
        app = build_app(self._make_agent(), app_type="agent")
        client = TestClient(app)
        r = client.post("/run", json={"data": "no prompt field"})
        assert r.status_code == 400
        assert "error" in r.json()

    def test_health(self):
        app = build_app(self._make_agent(), app_type="agent")
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200


# ───────────────────────────────────────────────────────────────────────
# CostTracker + BudgetGuard integration
# ───────────────────────────────────────────────────────────────────────


class TestCostBudgetIntegration:
    def test_tracker_feeds_guard(self):
        """CostTracker records can feed into BudgetGuard for cost control."""
        from synapsekit.observability.budget_guard import (
            BudgetExceededError,
            BudgetGuard,
            BudgetLimit,
        )
        from synapsekit.observability.cost_tracker import CostTracker

        tracker = CostTracker()
        guard = BudgetGuard(BudgetLimit(daily=0.01))

        with tracker.scope("pipeline"):
            rec = tracker.record("gpt-4o", 1000, 500, 200.0)
            guard.record_spend(rec.cost_usd)

        # The gpt-4o call should have cost enough to approach/exceed the tiny budget
        # Record another call that should trigger budget exceeded
        with tracker.scope("pipeline"):
            rec2 = tracker.record("gpt-4o", 1000, 500, 200.0)
            with pytest.raises(BudgetExceededError):
                guard.check_before(rec2.cost_usd)

    def test_eval_case_with_cost_tracking(self):
        """eval_case + CostTracker work together for eval with cost bounds."""
        from synapsekit.evaluation.decorators import eval_case
        from synapsekit.observability.cost_tracker import CostTracker

        tracker = CostTracker()

        @eval_case(min_score=0.7, max_cost_usd=1.0)
        def my_eval():
            with tracker.scope("eval"):
                tracker.record("gpt-4o-mini", 500, 200, 100.0)
            return {
                "score": 0.85,
                "cost_usd": tracker.total_cost_usd,
                "latency_ms": 100.0,
            }

        result = my_eval()
        assert result["score"] >= 0.7
        assert result["cost_usd"] > 0
        assert result["cost_usd"] < 1.0


# ───────────────────────────────────────────────────────────────────────
# PromptHub + PromptTemplate integration
# ───────────────────────────────────────────────────────────────────────


class TestPromptHubIntegration:
    def test_hub_returns_usable_template(self, tmp_path):
        from synapsekit.prompts.hub import PromptHub

        hub = PromptHub(hub_dir=tmp_path)
        hub.push("acme/summarize", "Summarize the following:\n\n{text}\n\nSummary:", version="v1")

        tpl = hub.pull("acme/summarize:v1")
        rendered = tpl.format(text="SynapseKit is a Python framework for RAG.")
        assert "SynapseKit is a Python framework for RAG." in rendered
        assert "Summarize the following:" in rendered

    def test_version_upgrade_workflow(self, tmp_path):
        from synapsekit.prompts.hub import PromptHub

        hub = PromptHub(hub_dir=tmp_path)
        hub.push("acme/qa", "Answer: {question}", version="v1")
        hub.push("acme/qa", "Please answer concisely: {question}", version="v2")

        # Latest should be v2
        tpl = hub.pull("acme/qa")
        assert "concisely" in tpl.format(question="test")

        # Can still pull v1
        tpl_v1 = hub.pull("acme/qa:v1")
        assert "Answer:" in tpl_v1.format(question="test")


# ───────────────────────────────────────────────────────────────────────
# Auto-detection integration
# ───────────────────────────────────────────────────────────────────────


class TestAutoDetection:
    def test_auto_detect_builds_correct_routes(self):
        """build_app auto-detects type and creates correct endpoints."""

        class RAG:
            pass

        class CompiledGraph:
            pass

        class FunctionCallingAgent:
            pass

        rag_app = build_app(RAG())
        graph_app = build_app(CompiledGraph())
        agent_app = build_app(FunctionCallingAgent())

        rag_routes = {r.path for r in rag_app.routes}
        graph_routes = {r.path for r in graph_app.routes}
        agent_routes = {r.path for r in agent_app.routes}

        assert "/query" in rag_routes
        assert "/run" in graph_routes
        assert "/stream" in graph_routes
        assert "/run" in agent_routes

        # All should have health
        for routes in [rag_routes, graph_routes, agent_routes]:
            assert "/health" in routes
