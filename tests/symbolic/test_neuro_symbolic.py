from __future__ import annotations

import inspect

import pytest

import synapsekit
from synapsekit.agents.base import ToolResult
from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.symbolic import (
    BaseSymbolicBackend,
    ConstraintExtractor,
    ConstraintSet,
    NeuroSymbolicAgent,
    ProofTrace,
    VerificationFailure,
    parse_constraint_response,
    verified_tool,
)
from synapsekit.symbolic.types import NeuroSymbolicResult


class FakeLLM(BaseLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig(model="fake", api_key="", provider="fake"))
        self.responses = responses
        self.prompts: list[str] = []

    async def generate(self, prompt: str, **kw) -> str:
        del kw
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return "translated answer"

    async def stream(self, prompt: str, **kw):
        del kw
        yield await self.generate(prompt)


class FakeBackend(BaseSymbolicBackend):
    name = "fake"
    language = "smtlib"

    def __init__(self, proofs: list[ProofTrace]) -> None:
        self.proofs = proofs
        self.solve_calls = 0
        self.verify_calls = 0

    async def solve(
        self,
        constraints: ConstraintSet,
        *,
        timeout_seconds: float | None = None,
    ) -> ProofTrace:
        del constraints, timeout_seconds
        self.solve_calls += 1
        if self.proofs:
            return self.proofs.pop(0)
        return ProofTrace(status="sat", model={"x": 1}, backend=self.name, verified=True)

    async def verify(
        self,
        constraints: ConstraintSet,
        answer: str,
        proof: ProofTrace,
        *,
        timeout_seconds: float | None = None,
    ) -> ProofTrace:
        del constraints, answer, timeout_seconds
        self.verify_calls += 1
        return proof


def test_parse_constraint_response_accepts_fenced_json() -> None:
    constraints = parse_constraint_response(
        """```json
        {"language": "smtlib", "source": "(declare-const x Int)", "metadata": {"k": "v"}}
        ```"""
    )

    assert constraints.language == "smtlib"
    assert constraints.source == "(declare-const x Int)"
    assert constraints.metadata == {"k": "v"}


def test_parse_constraint_response_rejects_missing_source() -> None:
    with pytest.raises(VerificationFailure):
        parse_constraint_response('{"language": "smtlib"}')


def test_public_exports_are_available_from_top_level() -> None:
    assert synapsekit.NeuroSymbolicAgent is NeuroSymbolicAgent
    assert synapsekit.ConstraintSet is ConstraintSet
    assert synapsekit.verified_tool is verified_tool


@pytest.mark.asyncio
async def test_agent_solves_and_returns_verified_result() -> None:
    llm = FakeLLM(
        [
            '{"language": "smtlib", "source": "(declare-const x Int)"}',
            "x = 1",
        ]
    )
    backend = FakeBackend([ProofTrace(status="sat", model={"x": 1}, backend="fake", verified=True)])
    agent = NeuroSymbolicAgent(llm=llm, verifier=backend)

    result = await agent.solve("find x")

    assert isinstance(result, NeuroSymbolicResult)
    assert isinstance(result.proof, ProofTrace)
    assert result.verified is True
    assert result.answer == "x = 1"
    assert result.proof.model == {"x": 1}
    assert result.attempts == 1
    assert backend.solve_calls == 1
    assert backend.verify_calls == 1


@pytest.mark.asyncio
async def test_agent_accepts_verified_empty_solver_model() -> None:
    llm = FakeLLM(['{"language": "smtlib", "source": "true"}', "constraints are satisfiable"])
    backend = FakeBackend([ProofTrace(status="sat", model={}, backend="fake", verified=True)])
    agent = NeuroSymbolicAgent(llm=llm, verifier=backend)

    result = await agent.solve("prove satisfiable")

    assert isinstance(result, NeuroSymbolicResult)
    assert isinstance(result.proof, ProofTrace)
    assert result.verified is True
    assert result.answer == "constraints are satisfiable"


@pytest.mark.asyncio
async def test_agent_retries_unverified_proposals() -> None:
    llm = FakeLLM(
        [
            '{"language": "smtlib", "source": "bad"}',
            '{"language": "smtlib", "source": "good"}',
            "good answer",
        ]
    )
    backend = FakeBackend(
        [
            ProofTrace(status="unknown", backend="fake", error="unknown"),
            ProofTrace(status="sat", model={"x": 2}, backend="fake", verified=True),
        ]
    )
    agent = NeuroSymbolicAgent(llm=llm, verifier=backend, max_proposals=2)

    result = await agent.solve("find x")

    assert isinstance(result, NeuroSymbolicResult)
    assert isinstance(result.proof, ProofTrace)
    assert result.verified is True
    assert result.answer == "good answer"
    assert result.attempts == 2
    assert "Previous verification failed" in llm.prompts[1]


@pytest.mark.asyncio
async def test_agent_reject_policy_raises_on_unverified_result() -> None:
    llm = FakeLLM(['{"language": "smtlib", "source": "bad"}', "bad answer"])
    backend = FakeBackend([ProofTrace(status="unsat", backend="fake", error="unsat")])
    agent = NeuroSymbolicAgent(llm=llm, verifier=backend, on_unverified="reject")

    with pytest.raises(VerificationFailure):
        await agent.solve("find x")


@pytest.mark.asyncio
async def test_verified_tool_returns_proof_metadata() -> None:
    backend = FakeBackend(
        [ProofTrace(status="sat", model={"ok": True}, backend="fake", verified=True)]
    )

    def build_constraints(kwargs: dict, output: str) -> ConstraintSet:
        return ConstraintSet(
            language="smtlib",
            source=f"; verify {kwargs['value']} -> {output}",
        )

    @verified_tool(backend, build_constraints)
    def double(value: int) -> int:
        return value * 2

    result = await double.run(value=4)

    assert isinstance(result, ToolResult)
    assert result.output == "8"
    assert result.metadata["proof"].is_verified is True


@pytest.mark.asyncio
async def test_constraint_extractor_uses_backend_language() -> None:
    llm = FakeLLM(['{"source": "(declare-const x Int)"}'])
    backend = FakeBackend([])
    extractor = ConstraintExtractor(llm)

    constraints = await extractor.extract("find x", backend=backend)

    assert constraints.language == "smtlib"
    assert "smtlib solver" in llm.prompts[0]


# ---------------------------------------------------------------------------
# New comprehensive tests
# ---------------------------------------------------------------------------


def test_verified_tool_preserves_async_coroutine_identity() -> None:
    """@verified_tool wrapping an async fn must yield an object whose .run is a coroutine."""
    backend = FakeBackend(
        [ProofTrace(status="sat", model={"ok": True}, backend="fake", verified=True)]
    )

    def build_constraints(kwargs: dict, output: str) -> ConstraintSet:
        return ConstraintSet(language="smtlib", source=f"; check {output}")

    @verified_tool(backend, build_constraints)
    async def async_fn(x: int) -> int:
        return x * 2

    assert inspect.iscoroutinefunction(async_fn.run)


@pytest.mark.asyncio
async def test_agent_handles_llm_returning_no_source() -> None:
    """When LLM returns JSON without 'source', agent should raise VerificationFailure."""
    llm = FakeLLM(['{"language": "smtlib"}'])
    backend = FakeBackend([])
    agent = NeuroSymbolicAgent(llm=llm, verifier=backend, on_unverified="reject", max_proposals=1)

    with pytest.raises(VerificationFailure):
        await agent.solve("find x")


@pytest.mark.asyncio
async def test_agent_flag_policy_returns_unverified_result() -> None:
    """on_unverified='flag' must return a result rather than raise, with verified=False."""
    llm = FakeLLM(['{"language": "smtlib", "source": "bad"}', "unverified answer"])
    backend = FakeBackend([ProofTrace(status="unsat", backend="fake", verified=False)])
    agent = NeuroSymbolicAgent(llm=llm, verifier=backend, on_unverified="flag", max_proposals=1)

    result = await agent.solve("find x")

    assert isinstance(result, NeuroSymbolicResult)
    assert isinstance(result.proof, ProofTrace)
    assert result.verified is False
    assert result.proof.is_verified is False


def test_agent_rejects_invalid_on_unverified_policy() -> None:
    with pytest.raises(ValueError, match="on_unverified"):
        NeuroSymbolicAgent(
            llm=FakeLLM([]),
            verifier=FakeBackend([]),
            on_unverified="wrong_policy",  # type: ignore[arg-type]
        )


def test_agent_rejects_zero_max_proposals() -> None:
    with pytest.raises(ValueError):
        NeuroSymbolicAgent(
            llm=FakeLLM([]),
            verifier=FakeBackend([]),
            max_proposals=0,
        )


def test_parse_constraint_response_rejects_bare_invalid_json() -> None:
    with pytest.raises(VerificationFailure, match=r"[Jj][Ss][Oo][Nn]"):
        parse_constraint_response("{not valid json at all}")


def test_z3_unsat_returns_unverified_trace() -> None:
    # importorskip only catches ImportError; a z3 whose *native* libz3 can't load
    # (e.g. an arch-mismatched wheel) raises z3types.Z3Exception on import/first
    # use. That is an environment problem, not a code defect, so skip explicitly —
    # on a correctly-installed z3 (CI) neither branch triggers and the test runs.
    try:
        import z3  # noqa: F401
    except Exception as exc:  # ImportError, or Z3Exception from a broken native lib
        pytest.skip(f"z3 unavailable in this environment: {exc}")
    import asyncio

    from synapsekit.symbolic.backends import Z3Backend

    result = asyncio.run(
        Z3Backend().solve(ConstraintSet(language="smtlib", source="(assert false)(check-sat)"))
    )
    assert result.status == "unsat"
    assert result.verified is False
