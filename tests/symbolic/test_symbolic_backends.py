from __future__ import annotations

import pytest

from synapsekit.symbolic import ConstraintSet, PrologBackend, SympyBackend


def test_z3_missing_dependency_error_message_is_actionable():
    """When z3 is absent, missing_optional_dependency returns an actionable message."""
    from synapsekit.symbolic.types import missing_optional_dependency

    err = missing_optional_dependency("z3-solver", "z3")
    assert "z3-solver" in str(err) or "z3" in str(err)
    assert isinstance(err, ImportError)


@pytest.mark.asyncio
async def test_sympy_backend_solves_expression_when_available() -> None:
    pytest.importorskip("sympy")

    result = await SympyBackend().solve(
        ConstraintSet(language="sympy", source="solve(Eq(Symbol('x') + 2, 5), Symbol('x'))")
    )

    assert result.status == "sat"
    assert result.verified is True
    assert result.model["result"] == "[3]"


@pytest.mark.asyncio
async def test_verify_rejects_answer_that_contradicts_solver(monkeypatch) -> None:
    # Regression for #911: verify() must compare the rendered answer to the
    # solver's numeric result, not just check that the model is satisfiable.
    pytest.importorskip("sympy")
    backend = SympyBackend()
    constraints = ConstraintSet(language="sympy", source="47*53")
    proof = await backend.solve(constraints)
    assert proof.model["result"] == "2491"

    correct = await backend.verify(constraints, "2491", proof)
    assert correct.is_verified is True

    wrong = await backend.verify(constraints, "2481", proof)
    assert wrong.is_verified is False
    assert "inconsistent" in (wrong.error or "")


@pytest.mark.asyncio
async def test_prolog_backend_missing_executable_is_actionable() -> None:
    backend = PrologBackend(executable="definitely-missing-swipl")

    with pytest.raises(ImportError, match="SWI-Prolog"):
        await backend.solve(
            ConstraintSet(
                language="prolog",
                source="parent(alice, bob).",
                objective="parent(alice, bob)",
            )
        )
