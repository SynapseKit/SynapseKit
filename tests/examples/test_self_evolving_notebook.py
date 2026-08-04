"""CI check that the self-evolving example notebook actually runs.

Executes ``examples/self_evolving_agent.ipynb`` in-process — every code cell, in
order, in one shared namespace, with top-level ``await`` supported — and asserts
the demonstrated behaviour holds (accuracy climbs, decoys are blocked, rollback
restores the earlier accuracy). This keeps the published notebook from rotting
without pulling a Jupyter kernel (ipykernel/pyzmq) into the test dependencies.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import sys
from itertools import pairwise
from pathlib import Path
from types import ModuleType

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "self_evolving_agent.ipynb"
_MODULE_NAME = "_self_evolving_notebook"


def _run_notebook(nb: nbformat.NotebookNode) -> dict[str, object]:
    """Execute every code cell in order in one shared namespace and return it.

    The cells are run inside a real module registered in ``sys.modules`` so that
    dataclass annotation resolution (which looks the class's module up there)
    works exactly as it would in a live kernel.
    """
    module = ModuleType(_MODULE_NAME)
    ns = module.__dict__
    sys.modules[_MODULE_NAME] = module
    loop = asyncio.new_event_loop()
    try:
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code" or not cell.source.strip():
                continue
            code = compile(
                cell.source,
                f"<cell {i}>",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
            result = eval(code, ns)  # trusted, first-party notebook
            if inspect.iscoroutine(result):
                loop.run_until_complete(result)
    finally:
        loop.close()
        sys.modules.pop(_MODULE_NAME, None)
    return ns


def test_self_evolving_notebook_runs_and_improves() -> None:
    assert NOTEBOOK.exists(), f"missing example notebook: {NOTEBOOK}"
    nb = nbformat.read(NOTEBOOK, as_version=4)

    ns = _run_notebook(nb)

    # The notebook builds these; assert the demonstrated outcomes actually hold.
    scores = ns["scores"]
    assert scores[0] <= 0.5, f"baseline should be fallible, got {scores[0]:.0%}"
    assert scores[-1] >= 0.99, f"final accuracy should be ~100%, got {scores[-1]:.0%}"
    for prev, curr in pairwise(scores):
        assert curr >= prev, f"accuracy regressed across cycles: {scores}"

    agent = ns["agent"]
    history = agent.evolution_history()
    blocked = [p for p in history if p.status == "blocked"]
    assert blocked, "no decoy patch was blocked by the eval gate"
    assert all(p.metadata.get("block_reason") for p in blocked)

    rolled_back = [p for p in history if p.status == "rolled_back"]
    assert rolled_back, "rollback cell did not append a rolled_back audit entry"

    # The rollback cell restored the pre-patch accuracy (100% -> 80%).
    assert ns["after_rollback"] < ns["before_rollback"]
