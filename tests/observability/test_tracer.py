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
