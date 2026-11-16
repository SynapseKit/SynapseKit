from __future__ import annotations

import asyncio
import importlib
import json

import pytest


@pytest.mark.asyncio
async def test_provider_error_trace_redacts_exception_message():
    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            raise RuntimeError("secret-api-key / private prompt / response body")

    result = await api().deliberate(
        Adapter(), "private prompt", n=1, max_output_tokens=8, total_output_budget=8
    )
    assert result.traces[0].status == "error"
    assert result.traces[0].error == "adapter request failed"
    assert "secret-api-key" not in repr(result)
    assert "private prompt" not in repr(result)


@pytest.mark.asyncio
async def test_openai_adapter_aclose_closes_injected_client():
    from types import SimpleNamespace

    closed = []

    async def close():
        closed.append(True)

    adapter = api().OpenAIChatAdapter(SimpleNamespace(max_retries=0, close=close), model="test")
    await adapter.aclose()
    assert closed == [True]


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_benchmark_cli_closes_factory_adapter_for_every_outcome(monkeypatch, capsys, outcome):
    import sys
    from types import SimpleNamespace

    bench = benchmark()
    closed = []

    class Adapter:
        async def aclose(self):
            await asyncio.sleep(0)
            closed.append(True)

    async def evaluate(*args, **kwargs):
        if outcome == "error":
            raise RuntimeError("evaluation failed")
        if outcome == "cancel":
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
        return {"ok": True}

    monkeypatch.setattr(
        bench.importlib, "import_module", lambda name: SimpleNamespace(create=Adapter)
    )
    monkeypatch.setattr(bench, "evaluate", evaluate)
    monkeypatch.setattr(sys, "argv", ["deliberation_eval.py", "--adapter", "fake:create"])
    if outcome == "success":
        bench.main()
        assert json.loads(capsys.readouterr().out) == {"ok": True}
    else:
        with pytest.raises(RuntimeError if outcome == "error" else asyncio.CancelledError):
            bench.main()
    assert closed == [True]


def test_deliberation_public_entrypoint_exists():
    module = importlib.import_module("synapsekit.llm.deliberation")
    assert callable(getattr(module, "deliberate", None))


def api():
    return importlib.import_module("synapsekit.llm.deliberation")


def completion(answer, confidence, cost=0.1):
    return api().Completion(json.dumps({"answer": answer, "confidence": confidence}), 10, 5, cost)


@pytest.mark.asyncio
async def test_huge_integer_confidence_becomes_invalid_trace_not_crash():
    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            return api().Completion(json.dumps({"answer": "ok", "confidence": 10**400}), 1, 1, 0.1)

    result = await api().deliberate(Adapter(), "q", n=1, max_output_tokens=8, total_output_budget=8)
    assert result.answer is None
    assert result.traces[0].status == "invalid"
    assert result.total_cost == 0.1


@pytest.mark.asyncio
async def test_huge_integer_cost_becomes_invalid_trace_not_crash():
    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            return api().Completion('{"answer":"ok","confidence":0.9}', 1, 1, 10**400)

    result = await api().deliberate(Adapter(), "q", n=1, max_output_tokens=8, total_output_budget=8)
    assert result.answer is None
    assert result.traces[0].status == "invalid"
    assert result.total_cost is None


def benchmark():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "deliberation_eval.py"
    spec = importlib.util.spec_from_file_location("deliberation_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["n", "short_cap"])
async def test_benchmark_invalid_positive_integer_rejected_before_baseline(field):
    calls = []

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            calls.append(prompt)
            return completion("4", 0.9)

    options = dict(n=3, short_cap=10, long_cap=50)
    options[field] = 0
    with pytest.raises(ValueError):
        await benchmark().evaluate(Adapter(), [{"prompt": "q", "answer": "4"}], **options)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cases",
    [
        [{"prompt": "q", "answer": "4"}, {"prompt": "bad"}],
        [{"prompt": "q", "answer": 4}],
        [{"prompt": 1, "answer": "4"}],
        None,
    ],
)
async def test_benchmark_rejects_invalid_fixture_before_any_request(cases):
    calls = []

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            calls.append(prompt)
            return completion("4", 0.9)

    with pytest.raises(ValueError):
        await benchmark().evaluate(Adapter(), cases, n=3, short_cap=10, long_cap=50)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 0),
        ("n", True),
        ("n", 1.5),
        ("short_cap", 0),
        ("short_cap", True),
        ("short_cap", 1.5),
        ("long_cap", 0),
        ("long_cap", True),
        ("long_cap", 1.5),
        ("early_exit_confidence", -0.1),
        ("early_exit_confidence", 1.1),
        ("early_exit_confidence", float("nan")),
        ("early_exit_confidence", float("inf")),
        ("early_exit_confidence", True),
        ("early_exit_confidence", "0.9"),
    ],
)
async def test_benchmark_invalid_configuration_never_calls_adapter(field, value):
    calls = []

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            calls.append(prompt)
            return completion("4", 0.9)

    options = dict(n=3, short_cap=10, long_cap=50, early_exit_confidence=None)
    options[field] = value
    with pytest.raises(ValueError):
        await benchmark().evaluate(Adapter(), [{"prompt": "q", "answer": "4"}], **options)
    assert calls == []


@pytest.mark.parametrize(
    "option,value",
    [
        ("--n", "0"),
        ("--short-cap", "0"),
        ("--long-cap", "0"),
        ("--early-exit-confidence", "nan"),
        ("--early-exit-confidence", "1.1"),
    ],
)
def test_benchmark_cli_invalid_configuration_never_loads_adapter(monkeypatch, option, value):
    import sys

    bench = benchmark()
    calls = []
    monkeypatch.setattr(bench.importlib, "import_module", lambda name: calls.append(name))
    monkeypatch.setattr(
        sys, "argv", ["deliberation_eval.py", "--adapter", "fake:create", option, value]
    )
    with pytest.raises(SystemExit):
        bench.main()
    assert calls == []


def test_benchmark_cli_passes_early_exit_option(monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    bench = benchmark()

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            return completion("4", 0.95)

    monkeypatch.setattr(
        bench.importlib, "import_module", lambda name: SimpleNamespace(create=Adapter)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["deliberation_eval.py", "--adapter", "fake:create", "--early-exit-confidence", "0.9"],
    )
    bench.main()
    report = json.loads(capsys.readouterr().out)
    assert report["early_exit_confidence"] == 0.9


def test_benchmark_cli_rejects_invalid_fixture_before_factory(monkeypatch, tmp_path):
    import sys

    fixture = tmp_path / "fixture.json"
    fixture.write_text('[{"prompt": "q"}]', encoding="utf-8")
    bench = benchmark()
    calls = []
    monkeypatch.setattr(bench.importlib, "import_module", lambda name: calls.append(name))
    monkeypatch.setattr(
        sys, "argv", ["deliberation_eval.py", "--adapter", "fake:create", "--fixture", str(fixture)]
    )
    with pytest.raises(SystemExit):
        bench.main()
    assert calls == []


@pytest.mark.asyncio
async def test_benchmark_opt_in_early_exit_reports_cancelled_unknown_usage():
    calls = []
    started = asyncio.Event()

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            calls.append(max_output_tokens)
            if max_output_tokens == 50:
                return completion("4", 0.9, 1.0)
            index = len(calls) - 1
            if index == 3:
                started.set()
            await started.wait()
            if index == 3:
                await asyncio.Event().wait()
            return completion("4", 0.95, 0.1)

    report = await benchmark().evaluate(
        Adapter(),
        [{"prompt": "q", "answer": "4"}],
        n=3,
        short_cap=10,
        long_cap=50,
        early_exit_confidence=0.9,
    )
    row = report["cases"][0]
    assert row["parallel_early_exit"] is True
    assert row["parallel_launched"] == 3
    assert row["parallel_cancelled"] == 1
    assert row["parallel_cost"] is None and report["parallel_total_cost"] is None
    assert row["parallel_output_tokens"] is None
    assert row["baseline_cost"] == 1.0
    assert report["parallel_early_exit_cases"] == 1
    assert report["parallel_total_launched"] == 3
    assert report["parallel_total_cancelled"] == 1


@pytest.mark.asyncio
async def test_concurrent_weighted_aggregation_and_cap():
    mod = api()
    all_started = asyncio.Event()
    calls = []

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            calls.append(max_output_tokens)
            i = len(calls) - 1
            if len(calls) == 3:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), 2)
            return [completion("A", 0.4), completion("A", 0.4), completion("B", 0.7)][i]

    result = await mod.deliberate(
        Adapter(),
        "question",
        n=3,
        max_output_tokens=8,
        total_output_budget=24,
        early_exit_confidence=None,
    )
    assert calls == [8, 8, 8]
    assert result.answer == "A"
    assert result.confidence == pytest.approx(0.4)
    assert result.total_cost == pytest.approx(0.3)
    assert result.total_input_tokens == 30
    assert result.total_output_tokens == 15
    assert len(result.traces) == 3
    assert not result.early_exit


@pytest.mark.asyncio
async def test_paired_benchmark_includes_every_launched_call():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "deliberation_eval.py"
    spec = importlib.util.spec_from_file_location("deliberation_eval", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            if max_output_tokens == 50:
                return completion("wrong", 0.7, 1.0)
            return completion("4", 0.9, 0.1)

    report = await bench.evaluate(
        Adapter(), [{"prompt": "2+2?", "answer": "4"}], n=3, short_cap=10, long_cap=50
    )
    assert report["cases"][0]["parallel_correct"] is True
    assert report["cases"][0]["baseline_correct"] is False
    assert report["parallel_total_cost"] == pytest.approx(0.3)
    assert report["baseline_total_cost"] == pytest.approx(1.0)
    assert report["early_exit_confidence"] is None
    assert report["parallel_early_exit_cases"] == 0
    assert report["parallel_total_launched"] == 3
    assert report["parallel_total_cancelled"] == 0


@pytest.mark.asyncio
async def test_paired_benchmark_preserves_unknown_cost():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "deliberation_eval.py"
    spec = importlib.util.spec_from_file_location("deliberation_eval", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            return api().Completion('{"answer":"4", "confidence": 0.9}')

    report = await bench.evaluate(
        Adapter(), [{"prompt": "2+2?", "answer": "4"}], n=2, short_cap=10, long_cap=50
    )
    assert report["parallel_total_cost"] is None
    assert report["baseline_total_cost"] is None


@pytest.mark.asyncio
async def test_early_exit_cancels_and_drains_and_does_not_claim_unknown_cost():
    mod = api()
    started = asyncio.Event()
    cleaned = asyncio.Event()
    calls = 0

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            nonlocal calls
            calls += 1
            i = calls
            if calls == 3:
                started.set()
            await started.wait()
            if i == 3:
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
            return completion("yes", 0.95)

    result = await mod.deliberate(
        Adapter(), "q", n=3, max_output_tokens=12, total_output_budget=36, early_exit_confidence=0.9
    )
    assert result.early_exit and result.answer == "yes"
    assert cleaned.is_set()
    assert result.total_cost is None and result.total_output_tokens is None
    assert sum(t.status == "cancelled" for t in result.traces) == 1


@pytest.mark.asyncio
async def test_two_wrong_votes_cannot_preempt_three_pending_votes(monkeypatch):
    mod = api()
    started = asyncio.Event()
    wrong_scored = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    original_score = mod._score

    def observe(traces):
        result = original_score(traces)
        if sum(t.answer == "wrong" for t in traces) == 2:
            wrong_scored.set()
        return result

    monkeypatch.setattr(mod, "_score", observe)

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            nonlocal calls
            i = calls
            calls += 1
            if calls == 5:
                started.set()
            await started.wait()
            if i >= 2:
                await release.wait()
            return completion("wrong" if i < 2 else "correct", 1.0)

    task = asyncio.create_task(
        mod.deliberate(
            Adapter(),
            "q",
            n=5,
            max_output_tokens=8,
            total_output_budget=40,
            early_exit_confidence=0.9,
        )
    )
    try:
        await asyncio.wait_for(wrong_scored.wait(), 2)
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        result = await task
    assert result.answer == "correct"
    assert len(result.traces) == 5


@pytest.mark.asyncio
async def test_equal_maximum_challenger_score_does_not_allow_early_exit(monkeypatch):
    mod = api()
    scored = asyncio.Event()
    release = asyncio.Event()
    original_score = mod._score
    calls = 0

    def observe(traces):
        result = original_score(traces)
        if len(traces) == 2:
            scored.set()
        return result

    monkeypatch.setattr(mod, "_score", observe)

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            nonlocal calls
            i = calls
            calls += 1
            if i >= 2:
                await release.wait()
            return completion("first" if i < 2 else "second", 1.0)

    task = asyncio.create_task(
        mod.deliberate(
            Adapter(),
            "q",
            n=4,
            max_output_tokens=8,
            total_output_budget=32,
            early_exit_confidence=0.9,
        )
    )
    try:
        await asyncio.wait_for(scored.wait(), 2)
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        result = await task
    assert not result.early_exit
    assert len(result.traces) == 4


@pytest.mark.asyncio
async def test_existing_challenger_is_included_in_margin(monkeypatch):
    mod = api()
    scored = asyncio.Event()
    release = asyncio.Event()
    original_score = mod._score
    calls = 0

    def observe(traces):
        result = original_score(traces)
        if len(traces) == 4:
            scored.set()
        return result

    monkeypatch.setattr(mod, "_score", observe)

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            nonlocal calls
            i = calls
            calls += 1
            if i >= 4:
                await release.wait()
            return completion("A" if i < 3 else "B", 0.7 if i < 3 else 1.0)

    task = asyncio.create_task(
        mod.deliberate(
            Adapter(),
            "q",
            n=6,
            max_output_tokens=8,
            total_output_budget=48,
            early_exit_confidence=0.6,
        )
    )
    try:
        await asyncio.wait_for(scored.wait(), 2)
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        result = await task
    assert result.answer == "B"
    assert not result.early_exit


@pytest.mark.asyncio
async def test_safe_margin_still_requires_confidence_threshold(monkeypatch):
    mod = api()
    scored = asyncio.Event()
    release = asyncio.Event()
    original_score = mod._score
    calls = 0

    def observe(traces):
        result = original_score(traces)
        if len(traces) == 2:
            scored.set()
        return result

    monkeypatch.setattr(mod, "_score", observe)

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            nonlocal calls
            i = calls
            calls += 1
            if i == 2:
                await release.wait()
            return completion("A", 0.8)

    task = asyncio.create_task(
        mod.deliberate(
            Adapter(),
            "q",
            n=3,
            max_output_tokens=8,
            total_output_budget=24,
            early_exit_confidence=0.9,
        )
    )
    try:
        await asyncio.wait_for(scored.wait(), 2)
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        result = await task
    assert result.answer == "A"
    assert not result.early_exit


@pytest.mark.asyncio
async def test_benchmark_uses_explicit_reasoning_baseline_and_preserves_trace():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "deliberation_eval.py"
    spec = importlib.util.spec_from_file_location("deliberation_eval", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    calls = []

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            calls.append((prompt, max_output_tokens))
            return api().Completion(
                json.dumps({"reasoning": "2 plus 2", "answer": "4", "confidence": 0.9}), 10, 5, 0.1
            )

    report = await bench.evaluate(
        Adapter(), [{"prompt": "2+2?", "answer": "4"}], n=2, short_cap=10, long_cap=50
    )
    baseline, parallel = calls[0], calls[1:]
    assert baseline[1] == 50 and all(cap == 10 for _, cap in parallel)
    assert "reasoning" in baseline[0].lower()
    assert "concisely" not in baseline[0].lower()
    assert all("concisely" in prompt.lower() for prompt, _ in parallel)
    assert all("2+2?" in prompt for prompt, _ in calls)
    assert report["cases"][0]["baseline_correct"] and report["cases"][0]["parallel_correct"]
    assert "2 plus 2" in report["cases"][0]["baseline_trace"]


@pytest.mark.asyncio
async def test_parent_cancellation_drains_all_children():
    mod = api()
    started = asyncio.Event()
    cleaned = []

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            i = len(cleaned)
            cleaned.append(False)
            if len(cleaned) == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned[i] = True

    task = asyncio.create_task(
        mod.deliberate(Adapter(), "q", n=2, max_output_tokens=4, total_output_budget=8)
    )
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned == [True, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.8"])
async def test_invalid_confidence_is_structured_failure(value):
    mod = api()

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            return completion("A", value)

    result = await mod.deliberate(Adapter(), "q", n=1, max_output_tokens=10, total_output_budget=10)
    assert result.answer is None
    assert result.traces[0].status == "invalid"
    assert result.total_cost == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_failure_and_unknown_usage_are_not_zero():
    mod = api()

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            raise RuntimeError("provider failed")

    result = await mod.deliberate(Adapter(), "q", n=2, max_output_tokens=10, total_output_budget=20)
    assert result.answer is None
    assert all(t.status == "error" for t in result.traces)
    assert result.total_cost is None and result.total_input_tokens is None


@pytest.mark.asyncio
async def test_budget_rejected_before_launch():
    mod = api()

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            pytest.fail("must not launch")

    with pytest.raises(ValueError, match="budget"):
        await mod.deliberate(Adapter(), "q", n=3, max_output_tokens=10, total_output_budget=29)


@pytest.mark.asyncio
async def test_early_exit_counts_cancel_suppressed_completion_cost():
    mod = api()
    started = asyncio.Event()
    calls = 0

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            nonlocal calls
            calls += 1
            i = calls
            if calls == 3:
                started.set()
            await started.wait()
            if i == 3:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return completion("late", 0.5, 0.4)
            return completion("yes", 0.95)

    result = await mod.deliberate(
        Adapter(), "q", n=3, max_output_tokens=8, total_output_budget=24, early_exit_confidence=0.9
    )
    assert result.early_exit
    assert result.total_cost == pytest.approx(0.6)
    assert len(result.traces) == 3


@pytest.mark.asyncio
async def test_early_exit_confidence_includes_late_completed_winner():
    mod = api()
    started = asyncio.Event()
    calls = 0

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            nonlocal calls
            calls += 1
            i = calls
            if calls == 3:
                started.set()
            await started.wait()
            if i == 3:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return completion("yes", 0.1)
            return completion("yes", 0.95)

    result = await mod.deliberate(
        Adapter(),
        "q",
        n=3,
        max_output_tokens=8,
        total_output_budget=24,
        early_exit_confidence=0.9,
    )
    assert result.early_exit
    assert result.answer == "yes"
    assert result.confidence == pytest.approx((0.95 + 0.95 + 0.1) / 3)
    assert all(t.status == "ok" for t in result.traces)


@pytest.mark.asyncio
async def test_openai_adapter_sets_provider_cap_and_accounts_usage():
    from types import SimpleNamespace

    mod = api()
    captured = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"answer":"ok","confidence":0.8}')
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=7),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()), max_retries=0)
    adapter = mod.OpenAIChatAdapter(
        client, model="test-model", input_price_per_token=0.01, output_price_per_token=0.02
    )
    result = await mod.deliberate(adapter, "q", n=1, max_output_tokens=12, total_output_budget=12)
    assert captured["max_completion_tokens"] == 12
    assert captured["model"] == "test-model"
    assert captured["stream"] is False
    assert result.answer == "ok"
    assert result.total_cost == pytest.approx(0.34)


def test_openai_adapter_rejects_automatic_retries():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="max_retries=0"):
        api().OpenAIChatAdapter(SimpleNamespace(max_retries=2), model="test")


@pytest.mark.asyncio
async def test_individual_provider_cancellation_is_a_failed_trace():
    mod = api()

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            raise asyncio.CancelledError

    result = await mod.deliberate(Adapter(), "q", n=1, max_output_tokens=10, total_output_budget=10)
    assert result.answer is None
    assert result.traces[0].status == "cancelled"
    assert result.total_cost is None


@pytest.mark.asyncio
async def test_cap_excess_is_invalid_and_usage_preserved():
    mod = api()

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            return mod.Completion('{"answer":"ok","confidence":0.8}', 2, 11, 0.2)

    result = await mod.deliberate(Adapter(), "q", n=1, max_output_tokens=10, total_output_budget=10)
    assert result.answer is None
    assert result.traces[0].status == "invalid"
    assert result.total_output_tokens == 11
    assert result.total_cost == 0.2


@pytest.mark.asyncio
async def test_no_false_early_exit_with_single_confident_vote():
    mod = api()

    class Adapter:
        async def complete(self, prompt, *, max_output_tokens):
            return completion("only", 1.0)

    result = await mod.deliberate(
        Adapter(), "q", n=1, max_output_tokens=10, total_output_budget=10, early_exit_confidence=0.9
    )
    assert not result.early_exit
