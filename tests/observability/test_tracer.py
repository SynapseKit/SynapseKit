"""Tests for TokenTracer."""

from __future__ import annotations

from synapsekit.observability.tracer import COST_TABLE, TokenTracer


class TestTokenTracer:
    def test_initial_summary_empty(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        s = tracer.summary()
        assert s["calls"] == 0
        assert s["total_tokens"] == 0
        assert s["estimated_cost_usd"] == 0.0

    def test_record_and_summary(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=100, output_tokens=50, latency_ms=200.0)
        s = tracer.summary()
        assert s["calls"] == 1
        assert s["total_input_tokens"] == 100
        assert s["total_output_tokens"] == 50
        assert s["total_tokens"] == 150
        assert s["total_latency_ms"] == 200.0

    def test_cost_calculation(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=1_000_000, output_tokens=1_000_000, latency_ms=0)
        s = tracer.summary()
        expected = (
            COST_TABLE["gpt-4o-mini"]["input"] * 1e6 + COST_TABLE["gpt-4o-mini"]["output"] * 1e6
        )
        assert abs(s["estimated_cost_usd"] - expected) < 1e-6

    def test_accumulates_multiple_records(self):
        tracer = TokenTracer(model="gpt-4o")
        tracer.record(100, 50, 100.0)
        tracer.record(200, 100, 200.0)
        s = tracer.summary()
        assert s["calls"] == 2
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 150

    def test_reset_clears_records(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(100, 50, 100.0)
        tracer.reset()
        s = tracer.summary()
        assert s["calls"] == 0

    def test_disabled_tracer_ignores_records(self):
        tracer = TokenTracer(model="gpt-4o-mini", enabled=False)
        tracer.record(1000, 500, 100.0)
        assert tracer.summary()["calls"] == 0

    def test_unknown_model_zero_cost(self):
        tracer = TokenTracer(model="unknown-model-xyz")
        tracer.record(1000, 500, 100.0)
        s = tracer.summary()
        assert s["estimated_cost_usd"] == 0.0

    def test_timer_helpers(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        t0 = tracer.start_timer()
        elapsed = tracer.elapsed_ms(t0)
        assert elapsed >= 0.0

    def test_quality_history_and_averages(self):
        tracer = TokenTracer(model="gpt-4o-mini")
        tracer.record(input_tokens=100, output_tokens=50, latency_ms=10.0)
        tracer.record_quality(faithfulness=0.8, relevancy=0.6, call_id=1)
        tracer.record(input_tokens=100, output_tokens=50, latency_ms=10.0)
        tracer.record_quality(faithfulness=1.0, relevancy=0.9, call_id=2)

        history = tracer.quality_history()
        assert len(history) == 2
        assert history[0]["call_id"] == 1
        assert history[1]["call_id"] == 2

        s = tracer.summary()
        assert s["avg_faithfulness"] == 0.9
        assert s["avg_relevancy"] == 0.75

    def test_quality_trend_improving_and_degrading(self):
        improving = TokenTracer(model="gpt-4o-mini", quality_window=4, quality_trend_threshold=0.01)
        for score in [0.2, 0.3, 0.8, 0.9]:
            improving.record(input_tokens=1, output_tokens=1, latency_ms=1.0)
            improving.record_quality(faithfulness=score, relevancy=score)
        assert improving.summary()["quality_trend"] == "improving"

        degrading = TokenTracer(model="gpt-4o-mini", quality_window=4, quality_trend_threshold=0.01)
        for score in [0.9, 0.8, 0.3, 0.2]:
            degrading.record(input_tokens=1, output_tokens=1, latency_ms=1.0)
            degrading.record_quality(faithfulness=score, relevancy=score)
        assert degrading.summary()["quality_trend"] == "degrading"
