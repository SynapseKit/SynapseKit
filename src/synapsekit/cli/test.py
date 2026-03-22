"""``synapsekit test`` — discover and run ``@eval_case``-decorated tests."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any


def _discover_eval_files(path: str) -> list[Path]:
    """Find eval_*.py and *_eval.py files."""
    root = Path(path)
    if root.is_file():
        return [root]
    files: list[Path] = []
    files.extend(root.rglob("eval_*.py"))
    files.extend(root.rglob("*_eval.py"))
    return sorted(set(files))


def _load_module(filepath: Path) -> Any:
    """Dynamically load a Python module from a file path."""
    module_name = filepath.stem
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _find_eval_cases(module: Any) -> list[tuple[str, Any]]:
    """Find all @eval_case-decorated functions in a module."""
    cases = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "_eval_case_meta"):
            cases.append((name, obj))
    return cases


def _check_thresholds(
    result: dict[str, Any],
    meta: Any,
    global_threshold: float,
) -> tuple[bool, list[str]]:
    """Check result against thresholds. Returns (passed, list of failure reasons)."""
    failures: list[str] = []
    score = result.get("score")

    # Check min_score from decorator or global threshold
    min_score = meta.min_score if meta.min_score is not None else global_threshold
    if score is not None and score < min_score:
        failures.append(f"score {score:.3f} < min {min_score:.3f}")

    # Check max_cost_usd from decorator
    cost = result.get("cost_usd")
    if meta.max_cost_usd is not None and cost is not None and cost > meta.max_cost_usd:
        failures.append(f"cost ${cost:.4f} > max ${meta.max_cost_usd:.4f}")

    # Check max_latency_ms from decorator
    latency = result.get("latency_ms")
    if meta.max_latency_ms is not None and latency is not None and latency > meta.max_latency_ms:
        failures.append(f"latency {latency:.0f}ms > max {meta.max_latency_ms:.0f}ms")

    return (len(failures) == 0, failures)


def run_test(args: Any) -> None:
    """Execute the ``synapsekit test`` command."""
    files = _discover_eval_files(args.path)

    if not files:
        print(f"No eval files found in '{args.path}'")
        sys.exit(0)

    results: list[dict[str, Any]] = []
    any_failed = False

    for filepath in files:
        mod = _load_module(filepath)
        if mod is None:
            continue

        cases = _find_eval_cases(mod)
        for name, fn in cases:
            meta = fn._eval_case_meta
            start = time.perf_counter()

            try:
                if inspect.iscoroutinefunction(fn):
                    import asyncio

                    result = asyncio.run(fn())
                else:
                    result = fn()
            except Exception as exc:
                result = {"score": 0.0, "error": str(exc)}

            elapsed_ms = (time.perf_counter() - start) * 1000

            if not isinstance(result, dict):
                result = {"score": float(result) if result is not None else 0.0}

            if "latency_ms" not in result:
                result["latency_ms"] = elapsed_ms

            passed, failures = _check_thresholds(result, meta, args.threshold)
            if not passed:
                any_failed = True

            results.append(
                {
                    "file": str(filepath),
                    "name": name,
                    "passed": passed,
                    "score": result.get("score"),
                    "cost_usd": result.get("cost_usd"),
                    "latency_ms": result.get("latency_ms"),
                    "failures": failures,
                    "tags": meta.tags,
                }
            )

    # Output
    if args.output_format == "json":
        print(json.dumps(results, indent=2, default=str))
    else:
        _print_table(results)

    # Snapshot / regression handling
    save_name = getattr(args, "save_snapshot", None)
    compare_baseline = getattr(args, "compare_baseline", None)
    fail_on_regression = getattr(args, "fail_on_regression", False)
    snapshot_dir = getattr(args, "snapshot_dir", ".synapsekit_evals")

    # Guard: only activate if the flags are actual strings (not Mock objects, etc.)
    if not isinstance(save_name, str):
        save_name = None
    if not isinstance(compare_baseline, str):
        compare_baseline = None
    if not isinstance(snapshot_dir, str):
        snapshot_dir = ".synapsekit_evals"

    if save_name or compare_baseline:
        from ..evaluation.regression import EvalRegression

        reg = EvalRegression(store_dir=snapshot_dir)

        if save_name:
            reg.save_snapshot(save_name, results)
            print(f"\nSnapshot saved: {save_name}")

        if compare_baseline:
            # Save current as a temp snapshot for comparison
            current_name = "__current__"
            reg.save_snapshot(current_name, results)
            report = reg.compare(compare_baseline, current_name)
            _print_regression_report(report)
            # Clean up temp snapshot
            temp_path = Path(snapshot_dir) / f"{current_name}.json"
            if temp_path.exists():
                temp_path.unlink()
            if fail_on_regression and report.has_regressions:
                sys.exit(1)

    if any_failed:
        sys.exit(1)


def _print_table(results: list[dict[str, Any]]) -> None:
    """Print results as a formatted table."""
    if not results:
        print("No eval cases found.")
        return

    print()
    print(f"{'Status':<8} {'Name':<40} {'Score':<10} {'Cost':<12} {'Latency':<12}")
    print("-" * 82)

    passed_count = 0
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        score = f"{r['score']:.3f}" if r["score"] is not None else "N/A"
        cost = f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "N/A"
        latency = f"{r['latency_ms']:.0f}ms" if r["latency_ms"] is not None else "N/A"
        print(f"{status:<8} {r['name']:<40} {score:<10} {cost:<12} {latency:<12}")
        if r["passed"]:
            passed_count += 1

        for failure in r.get("failures", []):
            print(f"         -> {failure}")

    print("-" * 82)
    total = len(results)
    print(f"{passed_count}/{total} passed")
    print()


def _print_regression_report(report: Any) -> None:
    """Print a regression comparison report."""
    print()
    print(f"Regression Report: {report.baseline_name} -> {report.current_name}")
    print("=" * 82)

    if not report.deltas:
        print("No comparable metrics found.")
        return

    print(f"{'Case':<30} {'Metric':<12} {'Baseline':<12} {'Current':<12} {'Delta':<12} {'Status'}")
    print("-" * 90)

    for d in report.deltas:
        status = "REGRESSED" if d.regressed else "OK"
        baseline = f"{d.baseline:.4f}" if d.baseline is not None else "N/A"
        current = f"{d.current:.4f}" if d.current is not None else "N/A"
        delta = f"{d.delta:+.4f}" if d.delta is not None else "N/A"
        print(f"{d.case_name:<30} {d.metric:<12} {baseline:<12} {current:<12} {delta:<12} {status}")

    print("-" * 90)
    if report.has_regressions:
        print("REGRESSIONS DETECTED")
    else:
        print("No regressions detected")
    print()
