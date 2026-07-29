from __future__ import annotations

import ast
import asyncio
import re
import tempfile
from pathlib import Path
from typing import Any

from .types import BaseSymbolicBackend, ConstraintSet, ProofTrace, missing_optional_dependency


class Z3Backend(BaseSymbolicBackend):
    """SMT backend powered by z3-solver."""

    name = "z3"
    language = "smtlib"

    async def solve(
        self,
        constraints: ConstraintSet,
        *,
        timeout_seconds: float | None = None,
    ) -> ProofTrace:
        try:
            import z3
        except ImportError:
            raise missing_optional_dependency("symbolic", "z3-solver") from None

        def _run() -> ProofTrace:
            solver = z3.Solver()
            if timeout_seconds is not None:
                solver.set("timeout", max(1, int(timeout_seconds * 1000)))
            solver.from_string(constraints.source)
            status = solver.check()
            if status == z3.sat:
                model = _z3_model_to_dict(solver.model())
                return ProofTrace(
                    status="sat",
                    model=model,
                    raw_output=str(solver.model()),
                    backend=self.name,
                    verified=True,
                )
            if status == z3.unsat:
                return ProofTrace(status="unsat", raw_output="unsat", backend=self.name)
            return ProofTrace(
                status="unknown",
                raw_output=solver.reason_unknown(),
                backend=self.name,
                error=solver.reason_unknown() or "z3 returned unknown",
            )

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            return ProofTrace(status="error", backend=self.name, error=str(exc), verified=False)


class SympyBackend(BaseSymbolicBackend):
    """Symbolic math backend powered by SymPy."""

    name = "sympy"
    language = "sympy"

    async def solve(
        self,
        constraints: ConstraintSet,
        *,
        timeout_seconds: float | None = None,
    ) -> ProofTrace:
        try:
            import sympy as sp
        except ImportError:
            raise missing_optional_dependency("symbolic", "sympy") from None

        def _run() -> ProofTrace:
            allowed: dict[str, Any] = {
                "Eq": sp.Eq,
                "Symbol": sp.Symbol,
                "symbols": sp.symbols,
                "solve": sp.solve,
                "simplify": sp.simplify,
                "factor": sp.factor,
                "expand": sp.expand,
            }
            _validate_sympy_expression(constraints.source, set(allowed))
            result = eval(constraints.source, {"__builtins__": {}}, allowed)
            return ProofTrace(
                status="sat",
                model={"result": str(result)},
                raw_output=str(result),
                backend=self.name,
                verified=True,
            )

        try:
            if timeout_seconds is None:
                return await asyncio.to_thread(_run)
            return await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout_seconds)
        except TimeoutError:
            return ProofTrace(
                status="unknown",
                backend=self.name,
                error="SymPy solve timed out.",
                verified=False,
            )
        except Exception as exc:
            return ProofTrace(status="error", backend=self.name, error=str(exc), verified=False)


class MiniZincBackend(BaseSymbolicBackend):
    """Constraint-programming backend powered by the minizinc Python package."""

    name = "minizinc"
    language = "minizinc"

    def __init__(self, solver_name: str = "gecode") -> None:
        self.solver_name = solver_name

    async def solve(
        self,
        constraints: ConstraintSet,
        *,
        timeout_seconds: float | None = None,
    ) -> ProofTrace:
        try:
            import minizinc
        except ImportError:
            raise missing_optional_dependency("symbolic", "minizinc") from None

        async def _run() -> ProofTrace:
            model = minizinc.Model()
            model.add_string(constraints.source)
            solver = minizinc.Solver.lookup(self.solver_name)
            instance = minizinc.Instance(solver, model)
            result = await instance.solve_async(timeout=timeout_seconds)
            solution = getattr(result, "solution", None)
            if solution is None:
                return ProofTrace(
                    status="unsat",
                    raw_output=str(result),
                    backend=self.name,
                    verified=False,
                )
            return ProofTrace(
                status="sat",
                model=_object_to_public_dict(solution),
                raw_output=str(result),
                backend=self.name,
                verified=True,
            )

        try:
            if timeout_seconds is None:
                return await _run()
            return await asyncio.wait_for(_run(), timeout=timeout_seconds + 1)
        except TimeoutError:
            return ProofTrace(
                status="unknown",
                backend=self.name,
                error="MiniZinc solve timed out.",
                verified=False,
            )
        except Exception as exc:
            return ProofTrace(status="error", backend=self.name, error=str(exc), verified=False)


class PrologBackend(BaseSymbolicBackend):
    """SWI-Prolog backend using the swipl executable."""

    name = "prolog"
    language = "prolog"

    def __init__(self, executable: str = "swipl") -> None:
        self.executable = executable

    async def solve(
        self,
        constraints: ConstraintSet,
        *,
        timeout_seconds: float | None = None,
    ) -> ProofTrace:
        query = constraints.objective or constraints.metadata.get("query") or "true"
        program = f"{constraints.source}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "query.pl"
            await asyncio.to_thread(path.write_text, program, encoding="utf-8")
            cmd = [
                self.executable,
                "-q",
                "-s",
                str(path),
                "-g",
                f"({query} -> writeln(true); writeln(false)), halt.",
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                raise ImportError(
                    "SWI-Prolog executable 'swipl' is required for PrologBackend."
                ) from None

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ProofTrace(
                    status="unknown",
                    backend=self.name,
                    error="Prolog solve timed out.",
                    verified=False,
                )

        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()
        if proc.returncode != 0:
            return ProofTrace(status="error", raw_output=out, backend=self.name, error=err)
        if out.lower().endswith("true"):
            return ProofTrace(
                status="sat",
                model={"query": query, "result": True},
                raw_output=out,
                backend=self.name,
                verified=True,
            )
        if out.lower().endswith("false"):
            return ProofTrace(status="unsat", raw_output=out, backend=self.name, verified=False)
        return ProofTrace(status="unknown", raw_output=out, backend=self.name, error=err or out)


def _z3_model_to_dict(model: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for decl in model.decls():
        values[decl.name()] = _coerce_solver_value(model[decl])
    return values


def _validate_sympy_expression(source: str, allowed_names: set[str]) -> None:
    tree = ast.parse(source, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.UnaryOp,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported SymPy expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unsupported SymPy name: {node.id}")


def _coerce_solver_value(value: Any) -> Any:
    text = str(value)
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if text in {"True", "False"}:
        return text == "True"
    return text


def _object_to_public_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return {str(key): value for key, value in obj.items()}
    values: dict[str, Any] = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            value = getattr(obj, key)
        except Exception:
            continue
        if not callable(value):
            values[key] = value
    return values or {"solution": str(obj)}
