"""Evaluation gates that bind approval to one exact diff bundle."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from .types import CommandResult

if TYPE_CHECKING:
    from .diff import DiffBundle
    from .environment import SandboxEnvironment


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


@dataclass(frozen=True, slots=True)
class EvalReceipt:
    """Immutable proof that an evaluator approved one exact diff."""

    passed: bool
    score: float
    threshold: float
    diff_sha256: str
    sandbox_id: str
    evaluator: str
    details: dict[str, Any]
    created_at: str
    receipt_hash: str

    @classmethod
    def create(
        cls,
        *,
        passed: bool,
        score: float,
        threshold: float,
        diff_sha256: str,
        sandbox_id: str,
        evaluator: str,
        details: Mapping[str, Any] | None = None,
    ) -> EvalReceipt:
        created_at = datetime.now(timezone.utc).isoformat()
        normalized_details = dict(details or {})
        body = {
            "passed": passed,
            "score": score,
            "threshold": threshold,
            "diff_sha256": diff_sha256,
            "sandbox_id": sandbox_id,
            "evaluator": evaluator,
            "details": normalized_details,
            "created_at": created_at,
        }
        return cls(
            passed=passed,
            score=score,
            threshold=threshold,
            diff_sha256=diff_sha256,
            sandbox_id=sandbox_id,
            evaluator=evaluator,
            details=normalized_details,
            created_at=created_at,
            receipt_hash=hashlib.sha256(_canonical(body)).hexdigest(),
        )


class EvalGate(Protocol):
    async def evaluate(self, diff: DiffBundle, environment: SandboxEnvironment) -> EvalReceipt: ...


def _result_fields(result: Any) -> tuple[bool, float, dict[str, Any]]:
    if isinstance(result, bool):
        return result, 1.0 if result else 0.0, {}
    if isinstance(result, (int, float)):
        value = float(result)
        return value > 0, value, {}
    if isinstance(result, Mapping):
        score = float(result.get("score", 1.0 if result.get("passed") else 0.0))
        passed = bool(result.get("passed", score > 0))
        details = {
            str(key): value for key, value in result.items() if key not in {"score", "passed"}
        }
        return passed, score, details
    passed = bool(getattr(result, "passed", False))
    score = float(getattr(result, "score", 1.0 if passed else 0.0))
    return passed, score, {"result": str(result)}


class CallableEvalGate:
    """Evaluate a diff with a sync or async callable supplied by the user."""

    def __init__(
        self,
        evaluator: Callable[[DiffBundle, SandboxEnvironment], Any],
        *,
        threshold: float = 0.5,
        name: str | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.threshold = threshold
        self.name: str = name or str(getattr(evaluator, "__name__", type(evaluator).__name__))

    async def evaluate(self, diff: DiffBundle, environment: SandboxEnvironment) -> EvalReceipt:
        result = self.evaluator(diff, environment)
        if inspect.isawaitable(result):
            result = await result
        passed, score, details = _result_fields(result)
        passed = passed and score >= self.threshold
        return EvalReceipt.create(
            passed=passed,
            score=score,
            threshold=self.threshold,
            diff_sha256=diff.digest,
            sandbox_id=environment.session_id,
            evaluator=self.name,
            details=details,
        )


class CommandEvalGate:
    """Run a fixed command inside the sandbox and gate on its exit code."""

    def __init__(self, command: Sequence[str], *, name: str = "command-eval") -> None:
        if not command:
            raise ValueError("Command eval gate requires a non-empty command.")
        self.command = tuple(command)
        self.name = name

    async def evaluate(self, diff: DiffBundle, environment: SandboxEnvironment) -> EvalReceipt:
        result: CommandResult = await environment.exec(self.command)
        return EvalReceipt.create(
            passed=result.ok,
            score=1.0 if result.ok else 0.0,
            threshold=1.0,
            diff_sha256=diff.digest,
            sandbox_id=environment.session_id,
            evaluator=self.name,
            details={
                "command": list(self.command),
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            },
        )


class EvalSuiteGate:
    """Adapter for an environment-aware evaluator object.

    Existing ``EvalSuite.score_prompt`` remains prompt-specific. This adapter
    intentionally requires an explicit ``evaluate`` or ``score_environment``
    method rather than passing filesystem diffs into a prompt API.
    """

    def __init__(self, suite: Any, *, threshold: float = 0.5) -> None:
        self.suite = suite
        self.threshold = threshold

    async def evaluate(self, diff: DiffBundle, environment: SandboxEnvironment) -> EvalReceipt:
        if not getattr(self.suite, "cases", [True]):
            return EvalReceipt.create(
                passed=False,
                score=0.0,
                threshold=self.threshold,
                diff_sha256=diff.digest,
                sandbox_id=environment.session_id,
                evaluator=type(self.suite).__name__,
                details={"error": "no eval cases found"},
            )
        method = getattr(self.suite, "evaluate", None) or getattr(
            self.suite, "score_environment", None
        )
        if method is None:
            raise TypeError(
                "EvalSuiteGate requires an environment-aware evaluate() or score_environment() method."
            )
        result = method(diff, environment)
        if inspect.isawaitable(result):
            result = await result
        passed, score, details = _result_fields(result)
        passed = passed and score >= self.threshold
        return EvalReceipt.create(
            passed=passed,
            score=score,
            threshold=self.threshold,
            diff_sha256=diff.digest,
            sandbox_id=environment.session_id,
            evaluator=type(self.suite).__name__,
            details=details,
        )
