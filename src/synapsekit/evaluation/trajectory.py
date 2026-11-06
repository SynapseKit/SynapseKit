"""Utilities for comparing ordered or unordered tool-call trajectories."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from synapsekit.audit.types import AuditRecord


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation name and its JSON-like argument value."""

    name: str
    arguments: Any


class MatchMode(str, Enum):
    """How expected calls relate to the actual trajectory."""

    STRICT = "strict"
    UNORDERED = "unordered"
    SUBSET = "subset"
    SUPERSET = "superset"


@dataclass(frozen=True)
class TrajectoryMismatch:
    """One detail explaining why two trajectories did not match."""

    kind: str
    expected_index: int | None = None
    actual_index: int | None = None
    expected: ToolCall | None = None
    actual: ToolCall | None = None


@dataclass(frozen=True)
class TrajectoryMatchResult:
    """Boolean outcome and deterministic details for a trajectory comparison."""

    matched: bool
    mismatches: tuple[TrajectoryMismatch, ...] = ()


class TrajectoryComparatorError(RuntimeError):
    """A custom argument comparator raised while matching a trajectory."""


def match_trajectory(
    expected: Iterable[ToolCall],
    actual: Iterable[ToolCall],
    *,
    mode: MatchMode | str = MatchMode.STRICT,
    argument_comparators: Mapping[str, Callable[[Any, Any], bool]] | None = None,
) -> TrajectoryMatchResult:
    """Compare tool calls using positional or multiset matching.

    Argument comparators are keyed by JSON Pointer paths in each call's
    ``arguments`` value. A comparator replaces structural equality at its
    selected node and receives ``(expected_value, actual_value)``. Audit
    records are adapted only when TOOL_CALL payloads provide explicit
    ``tool`` and ``input`` fields; the input value becomes the call's arguments.
    """
    match_mode = MatchMode(mode)
    comparators = _prepare_comparators(argument_comparators or {})
    expected_calls = list(expected)
    actual_calls = list(actual)

    if match_mode is MatchMode.STRICT:
        return _match_strict(expected_calls, actual_calls, comparators)
    return _match_multiset(expected_calls, actual_calls, match_mode, comparators)


def trajectory_from_audit_records(records: Iterable[AuditRecord]) -> list[ToolCall]:
    """Extract TOOL_CALL records with explicit tool names and input values.

    Other event kinds are ignored. A TOOL_CALL without the explicit
    ``tool``/``input`` call-start schema is rejected rather than inferring
    a tool name from a generic ``method`` field.
    """
    calls: list[ToolCall] = []
    error_message = (
        "TOOL_CALL audit record must include explicit 'tool' and 'input' fields "
        "in its call-start payload"
    )
    for record in records:
        if record.kind != "TOOL_CALL":
            continue
        payload = record.payload
        if not isinstance(payload, Mapping):
            raise ValueError(error_message)
        if any(key in payload for key in ("status", "output", "error")):
            raise ValueError(error_message)

        tool = payload.get("tool")
        if not isinstance(tool, str) or not tool.strip() or "input" not in payload:
            raise ValueError(error_message)
        calls.append(ToolCall(tool, payload["input"]))
    return calls


def _prepare_comparators(
    argument_comparators: Mapping[str, Callable[[Any, Any], bool]],
) -> dict[str, Callable[[Any, Any], bool]]:
    comparators: dict[str, Callable[[Any, Any], bool]] = {}
    for pointer, comparator in argument_comparators.items():
        if not isinstance(pointer, str):
            raise TypeError("argument comparator paths must be strings")
        _parse_json_pointer(pointer)
        if not callable(comparator):
            raise TypeError(f"argument comparator for {pointer!r} must be callable")
        comparators[pointer] = comparator
    return comparators


def _parse_json_pointer(pointer: str) -> tuple[str, ...]:
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON Pointer {pointer!r}: must start with '/' or be empty")

    tokens: list[str] = []
    for raw_token in pointer[1:].split("/"):
        token: list[str] = []
        index = 0
        while index < len(raw_token):
            char = raw_token[index]
            if char != "~":
                token.append(char)
                index += 1
                continue
            if index + 1 >= len(raw_token) or raw_token[index + 1] not in "01":
                raise ValueError(f"invalid JSON Pointer escape in {pointer!r}")
            token.append("~" if raw_token[index + 1] == "0" else "/")
            index += 2
        tokens.append("".join(token))
    return tuple(tokens)


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _arguments_equal(
    expected: Any,
    actual: Any,
    comparators: Mapping[str, Callable[[Any, Any], bool]],
    pointer: str = "",
) -> bool:
    pending: list[tuple[bool, Any, Any, str, tuple[int, int] | None]] = [
        (True, expected, actual, pointer, None)
    ]
    active_containers: set[tuple[int, int]] = set()

    while pending:
        visit, left, right, path, container_pair = pending.pop()
        if not visit:
            assert container_pair is not None
            active_containers.remove(container_pair)
            continue

        comparator = comparators.get(path)
        if comparator is not None:
            try:
                if not bool(comparator(left, right)):
                    return False
            except Exception as exc:
                raise TrajectoryComparatorError(
                    f"argument comparator at JSON Pointer {path!r} failed"
                ) from exc
            continue

        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if left.keys() != right.keys():
                return False
            pair = (id(left), id(right))
            if pair in active_containers:
                raise RecursionError("cyclic argument values cannot be compared")
            active_containers.add(pair)
            pending.append((False, left, right, path, pair))
            for key in reversed(list(left)):
                child_path = f"{path}/{_escape_pointer_token(str(key))}"
                pending.append((True, left[key], right[key], child_path, None))
            continue

        if isinstance(left, list | tuple) and isinstance(right, list | tuple):
            if len(left) != len(right):
                return False
            pair = (id(left), id(right))
            if pair in active_containers:
                raise RecursionError("cyclic argument values cannot be compared")
            active_containers.add(pair)
            pending.append((False, left, right, path, pair))
            for index in reversed(range(len(left))):
                pending.append((True, left[index], right[index], f"{path}/{index}", None))
            continue

        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                return False
        elif left != right:
            return False

    return True


def _match_strict(
    expected: list[ToolCall],
    actual: list[ToolCall],
    comparators: Mapping[str, Callable[[Any, Any], bool]],
) -> TrajectoryMatchResult:
    mismatches: list[TrajectoryMismatch] = []
    shared_count = min(len(expected), len(actual))
    for index in range(shared_count):
        expected_call = expected[index]
        actual_call = actual[index]
        if expected_call.name != actual_call.name:
            mismatches.append(
                TrajectoryMismatch("tool_name", index, index, expected_call, actual_call)
            )
        if not _arguments_equal(expected_call.arguments, actual_call.arguments, comparators):
            mismatches.append(
                TrajectoryMismatch("arguments", index, index, expected_call, actual_call)
            )

    mismatches.extend(
        TrajectoryMismatch("missing_call", index, None, expected[index], None)
        for index in range(shared_count, len(expected))
    )
    mismatches.extend(
        TrajectoryMismatch("unexpected_call", None, index, None, actual[index])
        for index in range(shared_count, len(actual))
    )
    return TrajectoryMatchResult(not mismatches, tuple(mismatches))


def _match_multiset(
    expected: list[ToolCall],
    actual: list[ToolCall],
    mode: MatchMode,
    comparators: Mapping[str, Callable[[Any, Any], bool]],
) -> TrajectoryMatchResult:
    if mode is MatchMode.SUPERSET:
        left = list(enumerate(actual))
        right = list(enumerate(expected))
        left_is_actual = True
    else:
        left = list(enumerate(expected))
        right = list(enumerate(actual))
        left_is_actual = False

    matched_right: dict[int, int] = {}
    matched_left: dict[int, int] = {}

    def calls_match(left_index: int, right_index: int) -> bool:
        left_call = left[left_index][1]
        right_call = right[right_index][1]
        expected_call, actual_call = (
            (right_call, left_call) if left_is_actual else (left_call, right_call)
        )
        return expected_call.name == actual_call.name and _arguments_equal(
            expected_call.arguments, actual_call.arguments, comparators
        )

    if not comparators:
        # Structural equality is an equivalence relation, so greedy matching
        # preserves the exact multiset result without exploring augmenting paths.
        for left_index in range(len(left)):
            for right_index in range(len(right)):
                if right_index in matched_right:
                    continue
                if calls_match(left_index, right_index):
                    matched_right[right_index] = left_index
                    matched_left[left_index] = right_index
                    break
    else:
        # Custom comparators need general bipartite matching: a later left call
        # may need to displace an earlier match. Walk the augmenting path
        # iteratively so trajectory size cannot consume Python call stack.
        for start_left_index in range(len(left)):
            visited_right: set[int] = set()
            visited_left = {start_left_index}
            parent_left: dict[int, int] = {}
            stack = [(start_left_index, 0)]
            free_right: int | None = None

            while stack:
                left_index, next_right_index = stack[-1]
                if next_right_index >= len(right):
                    stack.pop()
                    continue
                stack[-1] = (left_index, next_right_index + 1)
                if next_right_index in visited_right or not calls_match(
                    left_index, next_right_index
                ):
                    continue

                visited_right.add(next_right_index)
                parent_left[next_right_index] = left_index
                previous_left = matched_right.get(next_right_index)
                if previous_left is None:
                    free_right = next_right_index
                    break
                if previous_left not in visited_left:
                    visited_left.add(previous_left)
                    stack.append((previous_left, 0))

            if free_right is None:
                continue
            right_index = free_right
            while True:
                left_index = parent_left[right_index]
                previous_right = matched_left.get(left_index)
                matched_right[right_index] = left_index
                matched_left[left_index] = right_index
                if previous_right is None:
                    break
                right_index = previous_right

    matched_left_indices = set(matched_right.values())
    unmatched_left = [index for index in range(len(left)) if index not in matched_left_indices]
    unmatched_right = [index for index in range(len(right)) if index not in matched_right]

    if left_is_actual:
        unmatched_expected = [right[index][0] for index in unmatched_right]
        unmatched_actual = [left[index][0] for index in unmatched_left]
    else:
        unmatched_expected = [left[index][0] for index in unmatched_left]
        unmatched_actual = [right[index][0] for index in unmatched_right]

    mismatches: list[TrajectoryMismatch] = []
    if mode is MatchMode.SUBSET:
        mismatches.extend(
            TrajectoryMismatch("missing_call", index, None, expected[index], None)
            for index in unmatched_expected
        )
    elif mode is MatchMode.SUPERSET:
        mismatches.extend(
            TrajectoryMismatch("unexpected_call", None, index, None, actual[index])
            for index in unmatched_actual
        )
    else:
        paired_count = min(len(unmatched_expected), len(unmatched_actual))
        for offset in range(paired_count):
            expected_index = unmatched_expected[offset]
            actual_index = unmatched_actual[offset]
            expected_call = expected[expected_index]
            actual_call = actual[actual_index]
            kind = "tool_name" if expected_call.name != actual_call.name else "arguments"
            mismatches.append(
                TrajectoryMismatch(kind, expected_index, actual_index, expected_call, actual_call)
            )
        mismatches.extend(
            TrajectoryMismatch("missing_call", index, None, expected[index], None)
            for index in unmatched_expected[paired_count:]
        )
        mismatches.extend(
            TrajectoryMismatch("unexpected_call", None, index, None, actual[index])
            for index in unmatched_actual[paired_count:]
        )

    return TrajectoryMatchResult(not mismatches, tuple(mismatches))
