"""Neuro-symbolic reasoning benchmark for #733.

Quantifies the core claim — *LLM proposes, solver verifies* eliminates arithmetic
hallucination. We compare two ways of answering the same set of arithmetic/algebra
problems:

* **llm-only** — a model answers directly (simulated here by a `FakeLLM` that gets
  a realistic fraction of multi-digit arithmetic wrong, like real LLMs do);
* **neuro-symbolic** — the real :class:`NeuroSymbolicAgent` routes the problem
  through the real :class:`SympyBackend`, which computes and *verifies* the answer.

The extraction step (NL → formal expression) is deterministic here (a stub
extractor supplies the sympy expression) so the benchmark isolates the value of
the *solver*: with the same "model", answers that go through the solver are
guaranteed correct. Deterministic, offline, no API keys.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from synapsekit.llm.base import BaseLLM, LLMConfig
from synapsekit.symbolic.agent import NeuroSymbolicAgent
from synapsekit.symbolic.backends import SympyBackend
from synapsekit.symbolic.extractor import ConstraintExtractor
from synapsekit.symbolic.types import ConstraintSet

# (question, sympy expression, expected answer, the model's *direct* guess).
# ~40% of the direct guesses are wrong — the kind of slips LLMs make on
# multi-digit arithmetic. The sympy expression is always correct.
PROBLEMS: list[tuple[str, str, str, str]] = [
    ("17 * 23", "17*23", "391", "391"),
    ("47 * 53", "47*53", "2491", "2481"),  # wrong
    ("2**10 - 24", "2**10 - 24", "1000", "1000"),
    ("144 / 12 + 7", "144/12 + 7", "19", "19"),
    ("13 * 17 * 19", "13*17*19", "4199", "4200"),  # wrong
    ("999 + 888", "999 + 888", "1887", "1887"),
    ("7**4", "7**4", "2401", "2041"),  # wrong
    ("123 * 456", "123*456", "56088", "56088"),
    ("45 * 67 - 100", "45*67 - 100", "2915", "2900"),  # wrong
    ("sum 1..100 = n(n+1)/2", "100*101/2", "5050", "5050"),
    ("89 * 91", "89*91", "8099", "8091"),  # wrong
    ("256 * 4 + 3", "256*4 + 3", "1027", "1027"),
    ("31 * 37", "31*37", "1147", "1147"),
    ("1024 / 8 - 5", "1024/8 - 5", "123", "128"),  # wrong
    ("77 * 88", "77*88", "6776", "6776"),
    ("15**3", "15**3", "3375", "3125"),  # wrong
    ("321 + 654 + 987", "321 + 654 + 987", "1962", "1962"),
    ("64 * 64", "64*64", "4096", "4096"),
    ("199 * 3", "199*3", "597", "597"),
    ("53 * 47 - 11", "53*47 - 11", "2480", "2470"),  # wrong
]


class _FakeLLM(BaseLLM):
    def __init__(self, answer: str = "") -> None:
        super().__init__(LLMConfig(provider="fake", model="fake-1", api_key=""))
        self._answer = answer

    async def stream(self, prompt: str, **kw: object) -> AsyncGenerator[str, None]:
        yield self._answer

    async def generate(self, prompt: str, **kw: object) -> str:
        return self._answer


class _StubExtractor(ConstraintExtractor):
    """Deterministic NL→sympy extraction so the benchmark isolates the solver."""

    def __init__(self, expr: str) -> None:
        super().__init__(_FakeLLM())
        self._expr = expr

    async def extract(self, problem, *, backend, previous_error=None) -> ConstraintSet:  # type: ignore[override]
        return ConstraintSet(language="sympy", source=self._expr, metadata={"problem": problem})


def _num(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _correct(got: str, expected: str) -> bool:
    g, e = _num(got), _num(expected)
    return g is not None and e is not None and abs(g - e) < 1e-6


@dataclass
class Accuracy:
    label: str
    correct: int
    total: int

    @property
    def rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


async def _run_llm_only() -> Accuracy:
    correct = 0
    for _q, _expr, expected, direct in PROBLEMS:
        answer = await _FakeLLM(direct).generate("")
        correct += _correct(answer, expected)
    return Accuracy("llm-only", correct, len(PROBLEMS))


async def _run_neuro_symbolic() -> Accuracy:
    # The model still "proposes" its (sometimes wrong) direct answer, but the
    # answer we trust is the solver-grounded result the agent's backend computes
    # (proof.raw_output) — that's what makes the reasoning verified/correct.
    correct = 0
    for q, expr, expected, direct in PROBLEMS:
        agent = NeuroSymbolicAgent(
            llm=_FakeLLM(direct), verifier=SympyBackend(), extractor=_StubExtractor(expr)
        )
        result = await agent.solve(q)
        solver_answer = (result.proof.model or {}).get("result", result.proof.raw_output)
        correct += _correct(solver_answer, expected)
    return Accuracy("neuro-symbolic", correct, len(PROBLEMS))


async def compare() -> dict[str, Accuracy]:
    llm_only, neuro = await asyncio.gather(_run_llm_only(), _run_neuro_symbolic())
    return {"llm_only": llm_only, "neuro_symbolic": neuro}


def uplift(neuro: Accuracy, llm_only: Accuracy) -> float:
    return neuro.rate - llm_only.rate


if __name__ == "__main__":
    res = asyncio.run(compare())
    for a in res.values():
        print(f"{a.label:>15}: {a.correct}/{a.total} = {a.rate:.0%}")
    print(f"uplift: {uplift(res['neuro_symbolic'], res['llm_only']):.0%}")
