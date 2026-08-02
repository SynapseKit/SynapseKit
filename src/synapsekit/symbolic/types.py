from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

ConstraintLanguage = Literal["smtlib", "prolog", "minizinc", "sympy", "text"]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _first_number(text: object) -> float | None:
    if text is None:
        return None
    match = _NUMBER_RE.search(str(text).replace(",", ""))
    return float(match.group()) if match else None


def _answer_consistent_with_result(answer: str, result: object) -> bool:
    """True if a numeric solver result also appears (matching) in the answer.

    Only enforced when the solver produced a numeric result (e.g. SymPy math): a
    wrong rendered answer is rejected. For non-numeric results (Z3 models, Prolog,
    MiniZinc) we can't do a cheap numeric check, so we stay lenient rather than
    reject answers we cannot confidently disprove.
    """
    expected = _first_number(result)
    if expected is None:
        return True
    got = _first_number(answer)
    return got is not None and abs(got - expected) < 1e-6


SolverStatus = Literal["sat", "unsat", "unknown", "error"]
UnverifiedPolicy = Literal["retry", "reject", "flag"]


@dataclass
class ConstraintSet:
    """Formal constraints extracted from a natural-language task."""

    language: ConstraintLanguage
    source: str
    objective: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProofTrace:
    """Verifiable output returned by a symbolic backend."""

    status: SolverStatus
    model: dict[str, Any] = field(default_factory=dict)
    raw_output: str | None = None
    backend: str = "unknown"
    error: str | None = None
    verified: bool = False

    @property
    def is_verified(self) -> bool:
        return self.verified and self.status == "sat"


@dataclass
class NeuroSymbolicResult:
    """Final answer plus the solver trace used to verify it."""

    answer: str
    proof: ProofTrace
    verified: bool
    constraints: ConstraintSet
    attempts: int
    backend: str
    error: str | None = None


class VerificationFailure(RuntimeError):
    """Raised when a neuro-symbolic answer cannot be verified."""

    def __init__(
        self,
        message: str,
        *,
        proof: ProofTrace | None = None,
        constraints: ConstraintSet | None = None,
    ) -> None:
        super().__init__(message)
        self.proof = proof
        self.constraints = constraints


@runtime_checkable
class SymbolicBackend(Protocol):
    name: str
    language: ConstraintLanguage

    async def solve(
        self, constraints: ConstraintSet, *, timeout_seconds: float | None = None
    ) -> ProofTrace:
        """Solve or verify a formal constraint set."""
        ...

    async def verify(
        self,
        constraints: ConstraintSet,
        answer: str,
        proof: ProofTrace,
        *,
        timeout_seconds: float | None = None,
    ) -> ProofTrace:
        """Re-check that the rendered answer is consistent with the proof."""
        ...


class BaseSymbolicBackend:
    """Base class for solver backends with conservative round-trip verification."""

    name = "base"
    language: ConstraintLanguage = "text"

    async def solve(
        self, constraints: ConstraintSet, *, timeout_seconds: float | None = None
    ) -> ProofTrace:
        raise NotImplementedError

    async def verify(
        self,
        constraints: ConstraintSet,
        answer: str,
        proof: ProofTrace,
        *,
        timeout_seconds: float | None = None,
    ) -> ProofTrace:
        del constraints, timeout_seconds
        result = proof.model.get("result") if proof.model else None
        if result is None:
            result = proof.raw_output
        if (
            proof.status == "sat"
            and proof.verified
            and _answer_consistent_with_result(answer, result)
        ):
            return ProofTrace(
                status="sat",
                model=dict(proof.model),
                raw_output=proof.raw_output,
                backend=self.name,
                verified=True,
            )
        if proof.status == "sat" and proof.verified:
            # Solver found a model, but the rendered answer contradicts it.
            error = f"Rendered answer {answer!r} is inconsistent with the solver result {result!r}."
        else:
            error = proof.error or "Solver did not produce a satisfiable model."
        return ProofTrace(
            status=proof.status,
            model=dict(proof.model),
            raw_output=proof.raw_output,
            backend=self.name,
            error=error,
            verified=False,
        )


def missing_optional_dependency(extra: str, package: str) -> ImportError:
    return ImportError(
        f"{package} is required for this symbolic backend. "
        f"Install it with: pip install synapsekit[{extra}]"
    )
