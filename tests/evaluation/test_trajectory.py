from __future__ import annotations

import pytest

from synapsekit.evaluation.trajectory import (
    MatchMode,
    ToolCall,
    TrajectoryComparatorError,
    match_trajectory,
    trajectory_from_audit_records,
)


def _call(name: str, arguments: object = None) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def test_strict_mode_matches_exact_order_and_arguments():
    result = match_trajectory(
        [_call("search", {"query": "alpha"}), _call("open", {"id": 3})],
        [_call("search", {"query": "alpha"}), _call("open", {"id": 3})],
    )

    assert result.matched is True
    assert result.mismatches == ()


def test_strict_mode_reports_order_mismatch():
    result = match_trajectory(
        [_call("search"), _call("open")],
        [_call("open"), _call("search")],
        mode=MatchMode.STRICT,
    )

    assert result.matched is False
    assert [detail.kind for detail in result.mismatches] == ["tool_name", "tool_name"]


def test_unordered_mode_ignores_order_and_preserves_multiplicity():
    expected = [_call("search", {"q": 1}), _call("search", {"q": 1}), _call("open")]
    actual = [_call("open"), _call("search", {"q": 1}), _call("search", {"q": 1})]

    assert match_trajectory(expected, actual, mode="unordered").matched is True
    assert match_trajectory(expected, actual[:-1], mode="unordered").matched is False


def test_unordered_handles_1000_identical_calls_without_recursion():
    expected = [_call("search", {"q": "same"}) for _ in range(1000)]
    actual = [_call("search", {"q": "same"}) for _ in range(1000)]

    result = match_trajectory(expected, actual, mode="unordered")

    assert result.matched is True


def test_subset_mode_allows_additional_actual_calls_but_requires_multiplicity():
    expected = [_call("search"), _call("search")]

    assert (
        match_trajectory(
            expected,
            [_call("search"), _call("other"), _call("search")],
            mode="subset",
        ).matched
        is True
    )
    assert match_trajectory(expected, [_call("search")], mode="subset").matched is False


def test_superset_mode_allows_expected_calls_to_be_omitted():
    expected = [_call("search"), _call("open")]

    assert match_trajectory(expected, [_call("open")], mode="superset").matched is True
    assert match_trajectory(expected, [_call("other")], mode="superset").matched is False


def test_strict_mode_reports_tool_and_argument_mismatches():
    result = match_trajectory(
        [_call("search", {"query": "expected"})],
        [_call("open", {"query": "actual"})],
    )

    assert result.matched is False
    assert [detail.kind for detail in result.mismatches] == ["tool_name", "arguments"]
    assert all(detail.expected_index == 0 for detail in result.mismatches)
    assert all(detail.actual_index == 0 for detail in result.mismatches)


def test_strict_mode_reports_missing_and_extra_calls():
    missing = match_trajectory([_call("search")], [])
    extra = match_trajectory([], [_call("search")])

    assert [detail.kind for detail in missing.mismatches] == ["missing_call"]
    assert [detail.kind for detail in extra.mismatches] == ["unexpected_call"]


def test_json_pointer_comparator_overrides_only_the_selected_argument_value():
    result = match_trajectory(
        [_call("request", {"request": {"timestamp": 100, "body": "same"}})],
        [_call("request", {"request": {"timestamp": 101, "body": "same"}})],
        argument_comparators={
            "/request/timestamp": lambda expected, actual: abs(expected - actual) <= 1
        },
    )

    assert result.matched is True


def test_unordered_comparator_reassigns_wildcard_match_for_complete_assignment():
    result = match_trajectory(
        [_call("request", {"value": 1}), _call("request", {"value": 2})],
        [_call("request", {"value": "*"}), _call("request", {"value": 1})],
        mode=MatchMode.UNORDERED,
        argument_comparators={
            "/value": lambda expected, actual: actual == "*" or expected == actual
        },
    )

    assert result.matched is True
    assert result.mismatches == ()


def test_deep_argument_comparator_does_not_consume_python_call_stack():
    depth = 600
    expected: object = 0
    actual: object = 1
    for _ in range(depth):
        expected = [expected]
        actual = [actual]
    pointer = "/" + "/".join("0" for _ in range(depth))

    result = match_trajectory(
        [_call("nested", expected)],
        [_call("nested", actual)],
        argument_comparators={pointer: lambda left, right: left + 1 == right},
    )

    assert result.matched is True


def test_comparator_errors_are_reported_with_the_pointer():
    def broken_comparator(expected: object, actual: object) -> bool:
        raise RuntimeError("comparison failed")

    with pytest.raises(TrajectoryComparatorError, match="/request/timestamp") as exc_info:
        match_trajectory(
            [_call("request", {"request": {"timestamp": 100}})],
            [_call("request", {"request": {"timestamp": 101}})],
            argument_comparators={"/request/timestamp": broken_comparator},
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_audit_adapter_extracts_explicit_tool_records_in_order():
    from synapsekit.audit.trace import AuditTracer
    from synapsekit.audit.types import EventKind

    tracer = AuditTracer()
    tracer.record(EventKind.TOOL_CALL, {"tool": "search", "input": {"query": "alpha"}})
    tracer.record(EventKind.TOOL_CALL, {"tool": "open", "input": {"id": 3}})

    assert trajectory_from_audit_records(tracer.records) == [
        ToolCall("search", {"query": "alpha"}),
        ToolCall("open", {"id": 3}),
    ]


def test_audit_adapter_rejects_ambiguous_method_payload():
    from synapsekit.audit.trace import AuditTracer
    from synapsekit.audit.types import EventKind

    tracer = AuditTracer()
    tracer.record(EventKind.TOOL_CALL, {"method": "run", "args": ["alpha"], "kwargs": {}})

    with pytest.raises(ValueError):
        trajectory_from_audit_records(tracer.records)


def test_audit_adapter_rejects_normal_verifiable_agent_run_as_tool_call():
    from synapsekit.audit.trace import AuditTracer
    from synapsekit.audit.types import EventKind
    from synapsekit.audit.wrapper import VerifiableAgent

    class Agent:
        def run(self, prompt: str) -> str:
            return f"completed: {prompt}"

    tracer = AuditTracer()
    VerifiableAgent(Agent(), tracer=tracer).run("alpha")

    call_record = next(record for record in tracer.records if record.kind == EventKind.TOOL_CALL)
    assert call_record.payload["method"] == "run"

    with pytest.raises(ValueError, match=r"explicit.*tool.*input"):
        trajectory_from_audit_records(tracer.records)
