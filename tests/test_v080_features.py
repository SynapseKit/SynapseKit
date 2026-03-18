"""Tests for v0.8.0 features: evaluation metrics and observability (OTel, tracing UI)."""

from __future__ import annotations

import json
import os

import pytest

# ---------------------------------------------------------------------------
# Mock LLM helper
# ---------------------------------------------------------------------------


class MockLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    async def generate(self, prompt, **kw):
        r = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return r


# ===========================================================================
# Evaluation — MetricResult
# ===========================================================================


class TestMetricResult:
    def test_construction(self):
        from synapsekit.evaluation.base import MetricResult

        r = MetricResult(score=0.75, reason="good", details={"k": "v"})
        assert r.score == 0.75
        assert r.reason == "good"
        assert r.details == {"k": "v"}

    def test_defaults(self):
        from synapsekit.evaluation.base import MetricResult

        r = MetricResult(score=1.0)
        assert r.reason == ""
        assert r.details == {}

    def test_repr(self):
        from synapsekit.evaluation.base import MetricResult

        r = MetricResult(score=0.80, reason="ok")
        assert "0.80" in repr(r)
        assert "ok" in repr(r)


# ===========================================================================
# Evaluation — FaithfulnessMetric
# ===========================================================================


class TestFaithfulnessMetric:
    def test_import(self):
        from synapsekit.evaluation import FaithfulnessMetric

        assert FaithfulnessMetric.name == "faithfulness"

    async def test_all_supported(self):
        from synapsekit.evaluation import FaithfulnessMetric

        llm = MockLLM(["1. Python is a language", "YES"])
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(
            question="What is Python?",
            answer="Python is a language",
            contexts=["Python is a programming language."],
        )
        assert result.score == 1.0

    async def test_unsupported_claim(self):
        from synapsekit.evaluation import FaithfulnessMetric

        llm = MockLLM(["1. Claim A\n2. Claim B", "YES", "NO"])
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(
            question="q",
            answer="a",
            contexts=["ctx"],
        )
        assert result.score == pytest.approx(0.5)
        assert len(result.details["claims"]) == 2

    async def test_no_claims(self):
        from synapsekit.evaluation import FaithfulnessMetric

        llm = MockLLM(["NONE"])
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(
            question="q",
            answer="ok",
            contexts=["ctx"],
        )
        assert result.score == 1.0

    async def test_empty_extraction(self):
        from synapsekit.evaluation import FaithfulnessMetric

        llm = MockLLM(["no numbered lines here"])
        metric = FaithfulnessMetric(llm)
        result = await metric.evaluate(
            question="q",
            answer="a",
            contexts=["ctx"],
        )
        assert result.score == 1.0


# ===========================================================================
# Evaluation — RelevancyMetric
# ===========================================================================


class TestRelevancyMetric:
    def test_import(self):
        from synapsekit.evaluation import RelevancyMetric

        assert RelevancyMetric.name == "relevancy"

    async def test_all_relevant(self):
        from synapsekit.evaluation import RelevancyMetric

        llm = MockLLM(["YES"])
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(
            question="What is Python?",
            contexts=["Python is a language.", "Python was created by Guido."],
        )
        assert result.score == 1.0

    async def test_mixed_relevancy(self):
        from synapsekit.evaluation import RelevancyMetric

        llm = MockLLM(["YES", "NO"])
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(
            question="What is Python?",
            contexts=["Python info", "Unrelated stuff"],
        )
        assert result.score == pytest.approx(0.5)

    async def test_empty_contexts(self):
        from synapsekit.evaluation import RelevancyMetric

        llm = MockLLM(["YES"])
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(question="q", contexts=[])
        assert result.score == 0.0

    async def test_all_irrelevant(self):
        from synapsekit.evaluation import RelevancyMetric

        llm = MockLLM(["NO"])
        metric = RelevancyMetric(llm)
        result = await metric.evaluate(
            question="q",
            contexts=["irrelevant"],
        )
        assert result.score == 0.0


# ===========================================================================
# Evaluation — GroundednessMetric
# ===========================================================================


class TestGroundednessMetric:
    def test_import(self):
        from synapsekit.evaluation import GroundednessMetric

        assert GroundednessMetric.name == "groundedness"

    async def test_grounded(self):
        from synapsekit.evaluation import GroundednessMetric

        llm = MockLLM(["9"])
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(
            answer="Python was created by Guido.",
            contexts=["Python was created by Guido van Rossum."],
        )
        assert result.score == pytest.approx(0.9)

    async def test_empty_answer(self):
        from synapsekit.evaluation import GroundednessMetric

        llm = MockLLM(["5"])
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(answer="", contexts=["ctx"])
        assert result.score == 1.0

    async def test_no_contexts(self):
        from synapsekit.evaluation import GroundednessMetric

        llm = MockLLM(["5"])
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(answer="something", contexts=[])
        assert result.score == 0.0

    async def test_parse_failure_defaults(self):
        from synapsekit.evaluation import GroundednessMetric

        llm = MockLLM(["not a number"])
        metric = GroundednessMetric(llm)
        result = await metric.evaluate(answer="answer", contexts=["ctx"])
        assert result.score == 0.5


# ===========================================================================
# Evaluation — EvaluationResult
# ===========================================================================


class TestEvaluationResult:
    def test_construction(self):
        from synapsekit.evaluation.pipeline import EvaluationResult

        r = EvaluationResult(scores={"a": 0.8, "b": 0.6})
        assert r.scores == {"a": 0.8, "b": 0.6}

    def test_mean_score(self):
        from synapsekit.evaluation.pipeline import EvaluationResult

        r = EvaluationResult(scores={"a": 0.8, "b": 0.6})
        assert r.mean_score == pytest.approx(0.7)

    def test_mean_score_empty(self):
        from synapsekit.evaluation.pipeline import EvaluationResult

        r = EvaluationResult()
        assert r.mean_score == 0.0

    def test_repr(self):
        from synapsekit.evaluation.pipeline import EvaluationResult

        r = EvaluationResult(scores={"x": 0.9})
        text = repr(r)
        assert "0.90" in text


# ===========================================================================
# Evaluation — EvaluationPipeline
# ===========================================================================


class TestEvaluationPipeline:
    def test_import(self):
        from synapsekit.evaluation import EvaluationPipeline

        assert EvaluationPipeline is not None

    async def test_run_all_metrics(self):
        from synapsekit.evaluation import (
            EvaluationPipeline,
            FaithfulnessMetric,
            GroundednessMetric,
            RelevancyMetric,
        )

        # Faithfulness: "1. claim" then "YES"; Relevancy: "YES"; Groundedness: "8"
        llm = MockLLM(["1. claim", "YES", "YES", "8"])
        pipeline = EvaluationPipeline(
            metrics=[
                FaithfulnessMetric(llm),
                RelevancyMetric(llm),
                GroundednessMetric(llm),
            ]
        )
        result = await pipeline.evaluate(question="q", answer="a", contexts=["ctx"])
        assert "faithfulness" in result.scores
        assert "relevancy" in result.scores
        assert "groundedness" in result.scores
        assert 0.0 <= result.mean_score <= 1.0

    async def test_evaluate_batch(self):
        from synapsekit.evaluation import EvaluationPipeline, RelevancyMetric

        llm = MockLLM(["YES"])
        pipeline = EvaluationPipeline(metrics=[RelevancyMetric(llm)])
        results = await pipeline.evaluate_batch(
            [
                {"question": "q1", "contexts": ["c1"]},
                {"question": "q2", "contexts": ["c2"]},
            ]
        )
        assert len(results) == 2


# ===========================================================================
# Observability — Span
# ===========================================================================


class TestSpan:
    def test_construction(self):
        from synapsekit.observability.otel import Span

        s = Span("test", {"k": "v"})
        assert s.name == "test"
        assert s.attributes == {"k": "v"}
        assert s.status == "ok"

    def test_set_attribute(self):
        from synapsekit.observability.otel import Span

        s = Span("test")
        s.set_attribute("key", 42)
        assert s.attributes["key"] == 42

    def test_end(self):
        from synapsekit.observability.otel import Span

        s = Span("test")
        s.end()
        assert s.end_time is not None

    def test_duration(self):
        from synapsekit.observability.otel import Span

        s = Span("test")
        s.end()
        assert s.duration_ms >= 0

    def test_duration_running(self):
        from synapsekit.observability.otel import Span

        s = Span("test")
        assert s.duration_ms >= 0

    def test_to_dict(self):
        from synapsekit.observability.otel import Span

        s = Span("test", {"a": 1})
        s.end()
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["attributes"] == {"a": 1}
        assert "duration_ms" in d
        assert d["children"] == []

    def test_children(self):
        from synapsekit.observability.otel import Span

        parent = Span("parent")
        child = Span("child", parent=parent)
        assert len(parent.children) == 1
        assert parent.children[0] is child
        assert child.parent is parent

    def test_set_status(self):
        from synapsekit.observability.otel import Span

        s = Span("test")
        s.set_status("error")
        assert s.status == "error"


# ===========================================================================
# Observability — OTelExporter
# ===========================================================================


class TestOTelExporter:
    def test_import(self):
        from synapsekit.observability.otel import OTelExporter

        assert OTelExporter is not None

    def test_start_span(self):
        from synapsekit.observability.otel import OTelExporter

        exp = OTelExporter(service_name="test")
        span = exp.start_span("op", {"k": "v"})
        assert span.name == "op"
        assert len(exp.spans) == 1

    def test_end_span(self):
        from synapsekit.observability.otel import OTelExporter

        exp = OTelExporter()
        span = exp.start_span("op")
        exp.end_span(span)
        assert span.end_time is not None

    def test_export(self):
        from synapsekit.observability.otel import OTelExporter

        exp = OTelExporter()
        s = exp.start_span("root")
        exp.end_span(s)
        data = exp.export()
        assert len(data) == 1
        assert data[0]["name"] == "root"

    def test_clear(self):
        from synapsekit.observability.otel import OTelExporter

        exp = OTelExporter()
        exp.start_span("a")
        exp.clear()
        assert len(exp.spans) == 0

    def test_service_name(self):
        from synapsekit.observability.otel import OTelExporter

        exp = OTelExporter(service_name="my-app")
        assert exp._service_name == "my-app"

    def test_nested_spans(self):
        from synapsekit.observability.otel import OTelExporter

        exp = OTelExporter()
        parent = exp.start_span("parent")
        child = exp.start_span("child")
        exp.end_span(child)
        exp.end_span(parent)
        data = exp.export()
        assert len(data) == 1  # only root
        assert len(data[0]["children"]) == 1


# ===========================================================================
# Observability — TracingMiddleware
# ===========================================================================


class TestTracingMiddleware:
    def test_import(self):
        from synapsekit.observability.otel import TracingMiddleware

        assert TracingMiddleware is not None

    def test_trace_llm_wraps_generate(self):
        from synapsekit.observability.otel import OTelExporter, TracingMiddleware

        exp = OTelExporter()
        mw = TracingMiddleware(exp)
        llm = MockLLM(["hello"])
        traced = mw.trace_llm(llm)
        assert traced is llm  # mutates in place

    async def test_traced_call_records_span(self):
        from synapsekit.observability.otel import OTelExporter, TracingMiddleware

        exp = OTelExporter()
        mw = TracingMiddleware(exp)
        llm = MockLLM(["response"])
        mw.trace_llm(llm)
        result = await llm.generate("hello")
        assert result == "response"
        assert len(exp.spans) == 1
        assert exp.spans[0].name == "llm.generate"


# ===========================================================================
# Observability — TracingUI
# ===========================================================================


class TestTracingUI:
    def test_import(self):
        from synapsekit.observability.ui import TracingUI

        assert TracingUI is not None

    def test_render_html(self):
        from synapsekit.observability.otel import OTelExporter
        from synapsekit.observability.ui import TracingUI

        exp = OTelExporter()
        s = exp.start_span("op")
        exp.end_span(s)
        ui = TracingUI(exp)
        html = ui.render_html()
        assert "SynapseKit Traces" in html
        assert "op" in html

    def test_save_html(self, tmp_path):
        from synapsekit.observability.otel import OTelExporter
        from synapsekit.observability.ui import TracingUI

        exp = OTelExporter()
        exp.start_span("x").end()
        ui = TracingUI(exp)
        path = str(tmp_path / "traces.html")
        ui.save_html(path)
        assert os.path.exists(path)
        with open(path) as f:
            assert "SynapseKit" in f.read()

    def test_get_json(self):
        from synapsekit.observability.otel import OTelExporter
        from synapsekit.observability.ui import TracingUI

        exp = OTelExporter()
        s = exp.start_span("j")
        exp.end_span(s)
        ui = TracingUI(exp)
        data = json.loads(ui.get_json())
        assert isinstance(data, list)
        assert data[0]["name"] == "j"

    def test_render_empty(self):
        from synapsekit.observability.otel import OTelExporter
        from synapsekit.observability.ui import TracingUI

        exp = OTelExporter()
        ui = TracingUI(exp)
        html = ui.render_html()
        assert "SynapseKit Traces" in html


# ===========================================================================
# Top-level imports
# ===========================================================================


class TestTopLevelImports:
    def test_evaluation_imports(self):
        from synapsekit.evaluation import (
            EvaluationPipeline,
            EvaluationResult,
            FaithfulnessMetric,
            GroundednessMetric,
            MetricResult,
            RelevancyMetric,
        )

        assert all(
            [
                EvaluationPipeline,
                EvaluationResult,
                FaithfulnessMetric,
                GroundednessMetric,
                MetricResult,
                RelevancyMetric,
            ]
        )

    def test_observability_imports(self):
        from synapsekit.observability import (
            OTelExporter,
            Span,
            TracingMiddleware,
            TracingUI,
        )

        assert all([OTelExporter, Span, TracingMiddleware, TracingUI])
