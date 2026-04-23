"""Production-grade tests for TokenTracer quality metrics (auto-eval regression)."""

from __future__ import annotations

import pytest

from synapsekit.observability.tracer import TokenTracer


class TestTracerQualityNoneWhenNoData:
    """Regression: avg_faithfulness/avg_relevancy must be None when no quality data exists.

    Before fix: returned 0.0 which misled callers into thinking scores were recorded.
    """

    def test_avg_faithfulness_is_none_when_no_quality_records(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=10, output_tokens=5, latency_ms=100.0)
        summary = tracer.summary()
        assert summary["avg_faithfulness"] is None, (
            "avg_faithfulness must be None when no quality records exist, not 0.0"
        )

    def test_avg_relevancy_is_none_when_no_quality_records(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=10, output_tokens=5, latency_ms=100.0)
        summary = tracer.summary()
        assert summary["avg_relevancy"] is None, (
            "avg_relevancy must be None when no quality records exist, not 0.0"
        )

    def test_avg_faithfulness_none_with_zero_calls(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        summary = tracer.summary()
        assert summary["avg_faithfulness"] is None

    def test_avg_relevancy_none_with_zero_calls(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        summary = tracer.summary()
        assert summary["avg_relevancy"] is None


class TestTracerQualityWhenDataPresent:
    def test_avg_faithfulness_computed_when_records_exist(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=10, output_tokens=5, latency_ms=50.0)
        tracer.record_quality(faithfulness=0.8, call_id=1)
        tracer.record_quality(faithfulness=0.6, call_id=1)
        summary = tracer.summary()
        assert summary["avg_faithfulness"] == pytest.approx(0.7, abs=1e-4)

    def test_avg_relevancy_computed_when_records_exist(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=10, output_tokens=5, latency_ms=50.0)
        tracer.record_quality(relevancy=0.9, call_id=1)
        tracer.record_quality(relevancy=0.7, call_id=1)
        summary = tracer.summary()
        assert summary["avg_relevancy"] == pytest.approx(0.8, abs=1e-4)

    def test_faithfulness_only_relevancy_none(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=10, output_tokens=5, latency_ms=50.0)
        tracer.record_quality(faithfulness=0.5, call_id=1)
        summary = tracer.summary()
        assert summary["avg_faithfulness"] == pytest.approx(0.5, abs=1e-4)
        assert summary["avg_relevancy"] is None

    def test_relevancy_only_faithfulness_none(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=10, output_tokens=5, latency_ms=50.0)
        tracer.record_quality(relevancy=0.75, call_id=1)
        summary = tracer.summary()
        assert summary["avg_faithfulness"] is None
        assert summary["avg_relevancy"] == pytest.approx(0.75, abs=1e-4)

    def test_reset_clears_quality_records(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=10, output_tokens=5, latency_ms=50.0)
        tracer.record_quality(faithfulness=0.9, call_id=1)
        tracer.reset()
        summary = tracer.summary()
        assert summary["avg_faithfulness"] is None
        assert summary["avg_relevancy"] is None

    def test_multiple_records_averaged_correctly(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        for i in range(4):
            cid = tracer.record(input_tokens=5, output_tokens=5, latency_ms=10.0)
            tracer.record_quality(faithfulness=float(i) * 0.25, relevancy=1.0, call_id=cid)
        summary = tracer.summary()
        # faithfulness: 0.0, 0.25, 0.50, 0.75 → avg = 0.375
        assert summary["avg_faithfulness"] == pytest.approx(0.375, abs=1e-4)
        assert summary["avg_relevancy"] == pytest.approx(1.0, abs=1e-4)


class TestTracerQualityTrend:
    def test_stable_trend_with_insufficient_data(self):
        tracer = TokenTracer(model="gpt-4o-mini", quality_window=5)
        tracer.record(input_tokens=5, output_tokens=5, latency_ms=10.0)
        tracer.record_quality(faithfulness=0.5, call_id=1)
        summary = tracer.summary()
        assert summary["quality_trend"] == "stable"

    def test_improving_trend_detected(self):
        tracer = TokenTracer(model="gpt-4o-mini", quality_window=4, quality_trend_threshold=0.05)
        for val in [0.3, 0.4, 0.7, 0.8]:
            cid = tracer.record(input_tokens=5, output_tokens=5, latency_ms=10.0)
            tracer.record_quality(faithfulness=val, relevancy=val, call_id=cid)
        summary = tracer.summary()
        assert summary["quality_trend"] == "improving"

    def test_degrading_trend_detected(self):
        tracer = TokenTracer(model="gpt-4o-mini", quality_window=4, quality_trend_threshold=0.05)
        for val in [0.9, 0.8, 0.4, 0.3]:
            cid = tracer.record(input_tokens=5, output_tokens=5, latency_ms=10.0)
            tracer.record_quality(faithfulness=val, relevancy=val, call_id=cid)
        summary = tracer.summary()
        assert summary["quality_trend"] == "degrading"


class TestTracerSummaryReturnTypes:
    def test_summary_keys_present(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        summary = tracer.summary()
        expected_keys = {
            "model", "calls", "total_input_tokens", "total_output_tokens",
            "total_tokens", "total_latency_ms", "estimated_cost_usd",
            "avg_faithfulness", "avg_relevancy", "quality_trend",
        }
        assert expected_keys.issubset(summary.keys())

    def test_summary_avg_fields_are_float_or_none(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        summary = tracer.summary()
        for key in ("avg_faithfulness", "avg_relevancy"):
            val = summary[key]
            assert val is None or isinstance(val, float), f"{key}={val!r} must be float|None"

    def test_summary_avg_fields_are_float_when_data_present(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        cid = tracer.record(input_tokens=10, output_tokens=5, latency_ms=50.0)
        tracer.record_quality(faithfulness=0.8, relevancy=0.9, call_id=cid)
        summary = tracer.summary()
        assert isinstance(summary["avg_faithfulness"], float)
        assert isinstance(summary["avg_relevancy"], float)
