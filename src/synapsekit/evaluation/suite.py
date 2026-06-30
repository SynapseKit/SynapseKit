"""EvalSuite helpers for running ``@eval_case`` suites programmatically."""

from __future__ import annotations

import importlib.util
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .optimizer import PromptVariantRunner


@dataclass
class EvalSuiteResult:
    """Aggregate result for a prompt evaluated against an eval suite."""

    score: float
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    n_cases: int = 0
    passed: bool = False
    case_scores: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


class EvalSuite:
    """Programmatic wrapper around ``@eval_case`` functions.

    ``EvalSuite`` mirrors the discovery convention used by ``synapsekit test``
    and ``PromptOptimizer``: files named ``eval_*.py`` or ``*_eval.py`` are
    scanned for functions decorated with ``@eval_case``. Cases that accept a
    ``prompt`` keyword argument receive the candidate prompt, which makes the
    suite suitable for prompt-evolution gates.
    """

    def __init__(
        self,
        cases: list[tuple[str, Any]],
        *,
        metric: str = "score",
        threshold: float | None = None,
        source_path: str | None = None,
    ) -> None:
        self.cases = cases
        self.metric = metric
        self.threshold = threshold
        self.source_path = source_path

    @classmethod
    def from_decorators(
        cls,
        path: str = ".",
        *,
        metric: str = "score",
        threshold: float | None = None,
    ) -> EvalSuite:
        """Load eval cases from a file or directory using decorator discovery."""
        return cls(
            _load_eval_cases(path),
            metric=metric,
            threshold=threshold,
            source_path=path,
        )

    @classmethod
    def from_cases(
        cls,
        cases: list[tuple[str, Any]] | list[Any],
        *,
        metric: str = "score",
        threshold: float | None = None,
    ) -> EvalSuite:
        """Build a suite directly from callables, useful in tests and notebooks."""
        named: list[tuple[str, Any]] = []
        for index, case in enumerate(cases):
            if isinstance(case, tuple):
                named.append((str(case[0]), case[1]))
            else:
                named.append((getattr(case, "__name__", f"case_{index}"), case))
        return cls(named, metric=metric, threshold=threshold)

    @property
    def has_cases(self) -> bool:
        return bool(self.cases)

    async def score_prompt(self, prompt: str) -> EvalSuiteResult:
        """Run all cases for *prompt* and return aggregate metrics."""
        if not self.cases:
            return EvalSuiteResult(score=0.0, passed=False, failures=["no eval cases found"])

        start = time.perf_counter()
        runner = PromptVariantRunner(self.cases, self.metric)
        candidate = await runner.run_variant(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000
        case_scores = []
        if candidate.metadata and isinstance(candidate.metadata.get("scores"), list):
            case_scores = [float(v) for v in candidate.metadata["scores"]]

        threshold = self.threshold
        passed = True if threshold is None else candidate.score >= threshold
        failures: list[str] = []
        if threshold is not None and candidate.score < threshold:
            failures.append(f"{self.metric} {candidate.score:.3f} < min {threshold:.3f}")

        return EvalSuiteResult(
            score=candidate.score,
            cost_usd=candidate.cost_usd,
            latency_ms=elapsed_ms,
            n_cases=len(self.cases),
            passed=passed,
            case_scores=case_scores,
            failures=failures,
        )


def _discover_eval_files(path: str) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    files: list[Path] = []
    files.extend(root.rglob("eval_*.py"))
    files.extend(root.rglob("*_eval.py"))
    return sorted(set(files))


def _load_eval_cases(path: str) -> list[tuple[str, Any]]:
    cases: list[tuple[str, Any]] = []
    for filepath in _discover_eval_files(path):
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception:
            continue

        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and hasattr(obj, "_eval_case_meta"):
                cases.append((name, obj))

    return _dedupe_wrapped_cases(cases)


def _dedupe_wrapped_cases(cases: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Drop duplicate decorator wrappers while preserving stable ordering."""
    seen: set[tuple[str, int]] = set()
    deduped: list[tuple[str, Any]] = []
    for name, fn in cases:
        target = inspect.unwrap(fn)
        key = (name, id(target))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, fn))
    return deduped
