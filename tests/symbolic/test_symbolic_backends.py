from __future__ import annotations

import builtins

import pytest

from synapsekit.symbolic import ConstraintSet, PrologBackend, SympyBackend, Z3Backend


@pytest.mark.asyncio
async def test_z3_backend_reports_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "z3":
            raise ImportError("no z3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"synapsekit\[symbolic\]"):
        await Z3Backend().solve(ConstraintSet(language="smtlib", source="(check-sat)"))


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
