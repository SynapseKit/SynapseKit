from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import BaseLLM
from .extractor import ConstraintExtractor
from .types import (
    ConstraintSet,
    NeuroSymbolicResult,
    ProofTrace,
    SymbolicBackend,
    UnverifiedPolicy,
    VerificationFailure,
)

_TRANSLATION_PROMPT = """\
Render this verified solver model as a concise answer to the original problem.

Original problem:
{problem}

Constraint language: {language}
Solver status: {status}
Solver model:
{model}

Return only the final answer.
"""


@dataclass
class NeuroSymbolicAgent:
    """Agent that lets an LLM propose constraints and a solver verify them."""

    llm: BaseLLM
    verifier: SymbolicBackend
    on_unverified: UnverifiedPolicy = "retry"
    max_proposals: int = 3
    timeout_seconds: float | None = 10.0
    extractor: ConstraintExtractor | None = None

    def __post_init__(self) -> None:
        if self.on_unverified not in {"retry", "reject", "flag"}:
            raise ValueError("on_unverified must be 'retry', 'reject', or 'flag'.")
        if self.max_proposals < 1:
            raise ValueError("max_proposals must be >= 1.")
        if self.extractor is None:
            self.extractor = ConstraintExtractor(self.llm)

    async def solve(self, problem: str) -> NeuroSymbolicResult:
        last_error: str | None = None
        last_constraints: ConstraintSet | None = None
        last_proof: ProofTrace | None = None

        for attempt in range(1, self.max_proposals + 1):
            try:
                assert self.extractor is not None
                constraints = await self.extractor.extract(
                    problem,
                    backend=self.verifier,
                    previous_error=last_error,
                )
                last_constraints = constraints
                proof = await self.verifier.solve(
                    constraints,
                    timeout_seconds=self.timeout_seconds,
                )
                last_proof = proof
                answer = await self._translate(problem, constraints, proof)
                verification = await self.verifier.verify(
                    constraints,
                    answer,
                    proof,
                    timeout_seconds=self.timeout_seconds,
                )

                if verification.is_verified:
                    return NeuroSymbolicResult(
                        answer=answer,
                        proof=verification,
                        verified=True,
                        constraints=constraints,
                        attempts=attempt,
                        backend=self.verifier.name,
                    )

                last_error = verification.error or f"Solver returned {verification.status}."
                last_proof = verification
                if self.on_unverified != "retry":
                    return self._handle_unverified(
                        answer=answer,
                        constraints=constraints,
                        proof=verification,
                        attempts=attempt,
                        error=last_error,
                    )
            except VerificationFailure as exc:
                last_error = str(exc)
                if exc.constraints is not None:
                    last_constraints = exc.constraints
                if exc.proof is not None:
                    last_proof = exc.proof
                if self.on_unverified != "retry":
                    raise

        fallback_constraints = last_constraints or ConstraintSet(
            language=self.verifier.language,
            source="",
            metadata={"problem": problem},
        )
        fallback_proof = last_proof or ProofTrace(
            status="error",
            backend=self.verifier.name,
            error=last_error or "No verification attempt completed.",
            verified=False,
        )
        return self._handle_unverified(
            answer="",
            constraints=fallback_constraints,
            proof=fallback_proof,
            attempts=self.max_proposals,
            error=last_error,
        )

    async def _translate(
        self,
        problem: str,
        constraints: ConstraintSet,
        proof: ProofTrace,
    ) -> str:
        if proof.status != "sat":
            return proof.raw_output or proof.error or proof.status
        prompt = _TRANSLATION_PROMPT.format(
            problem=problem,
            language=constraints.language,
            status=proof.status,
            model=proof.model,
        )
        return (await self.llm.generate(prompt)).strip()

    def _handle_unverified(
        self,
        *,
        answer: str,
        constraints: ConstraintSet,
        proof: ProofTrace,
        attempts: int,
        error: str | None,
    ) -> NeuroSymbolicResult:
        if self.on_unverified == "reject":
            raise VerificationFailure(
                error or "Could not verify neuro-symbolic answer.",
                proof=proof,
                constraints=constraints,
            )
        return NeuroSymbolicResult(
            answer=answer,
            proof=proof,
            verified=False,
            constraints=constraints,
            attempts=attempts,
            backend=self.verifier.name,
            error=error,
        )
