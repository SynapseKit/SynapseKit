"""
Regression tests for @eval_case decorator and CLI runner.

These tests exist specifically to catch the class of bug where async
@eval_case functions are not correctly awaited by the CLI runner.

Root cause of v1.5.1 → v1.5.2 fix:
  The decorator used a sync wrapper for all functions, causing
  inspect.iscoroutinefunction() to return False for async functions.
  cli/test.py uses iscoroutinefunction() to decide whether to call
  asyncio.run(). Async functions were therefore never awaited —
  the raw coroutine object was passed to float(), raising TypeError.

Every test in this file must remain green for any release.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Decorator: iscoroutinefunction preservation
# ──────────────────────────────────────────────────────────────────────────────

class TestEvalCaseDecoratorAsync:
    """Verify the decorator preserves coroutine identity for async functions."""

    def test_async_fn_is_coroutinefunction_after_decoration(self):
        """inspect.iscoroutinefunction must return True for decorated async fns.

        This is the exact check that cli/test.py uses. If this fails, async
        eval cases will never be awaited and the CLI will raise TypeError.
        """
        from synapsekit.evaluation.decorators import eval_case

        @eval_case(min_score=0.8)
        async def eval_something():
            return {"score": 0.9}

        assert inspect.iscoroutinefunction(eval_something), (
            "Decorated async function must satisfy inspect.iscoroutinefunction(). "
            "If this fails, cli/test.py will call fn() without asyncio.run() and "
            "pass a coroutine object to float(), raising TypeError."
        )

    def test_sync_fn_is_not_coroutinefunction_after_decoration(self):
        """Sync functions must not be falsely identified as coroutines."""
        from synapsekit.evaluation.decorators import eval_case

        @eval_case(min_score=0.8)
        def eval_sync():
            return {"score": 0.9}

        assert not inspect.iscoroutinefunction(eval_sync)

    def test_async_fn_metadata_preserved(self):
        """_eval_case_meta must be present on the async wrapper."""
        from synapsekit.evaluation.decorators import eval_case

        @eval_case(min_score=0.75, max_cost_usd=0.02, tags=["rag"])
        async def eval_with_meta():
            return {"score": 0.8}

        assert hasattr(eval_with_meta, "_eval_case_meta")
        meta = eval_with_meta._eval_case_meta
        assert meta.min_score == 0.75
        assert meta.max_cost_usd == 0.02
        assert meta.tags == ["rag"]

    def test_async_fn_name_preserved(self):
        """functools.wraps must preserve __name__ on the async wrapper."""
        from synapsekit.evaluation.decorators import eval_case

        @eval_case(min_score=0.8)
        async def eval_my_pipeline():
            return {"score": 0.9}

        assert eval_my_pipeline.__name__ == "eval_my_pipeline"

    def test_async_fn_returns_correct_result(self):
        """Awaiting the decorated async function must return the original dict."""
        from synapsekit.evaluation.decorators import eval_case

        @eval_case(min_score=0.5)
        async def eval_returns_dict():
            return {"score": 0.85, "cost_usd": 0.005}

        result = asyncio.run(eval_returns_dict())
        assert result == {"score": 0.85, "cost_usd": 0.005}

    def test_async_fn_exception_propagates(self):
        """Exceptions inside an async eval case must propagate normally."""
        from synapsekit.evaluation.decorators import eval_case

        @eval_case(min_score=0.5)
        async def eval_raises():
            raise ValueError("pipeline failed")

        with pytest.raises(ValueError, match="pipeline failed"):
            asyncio.run(eval_raises())


# ──────────────────────────────────────────────────────────────────────────────
# CLI runner: async eval cases run end-to-end through run_test()
# ──────────────────────────────────────────────────────────────────────────────

class TestRunTestAsync:
    """End-to-end tests of cli/test.py with async eval case files.

    These write actual .py files to a temp directory and call run_test(),
    exactly as the GitHub Action does. This is the integration layer that
    was missing before v1.5.2.
    """

    def _make_args(self, path: str, threshold: float = 0.7, fmt: str = "json") -> Mock:
        args = Mock()
        args.path = path
        args.threshold = threshold
        args.output_format = fmt
        args.save_snapshot = None
        args.compare_baseline = None
        args.fail_on_regression = False
        args.snapshot_dir = ".synapsekit_evals"
        return args

    def test_async_eval_case_passes(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """An async @eval_case that returns score >= threshold must pass."""
        from synapsekit.cli.test import run_test

        (tmp_path / "eval_async.py").write_text(textwrap.dedent("""
            from synapsekit.evaluation.decorators import eval_case

            @eval_case(min_score=0.80)
            async def eval_passing_async():
                return {"score": 0.90, "cost_usd": 0.001}
        """))

        run_test(self._make_args(str(tmp_path)))
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 1
        assert data[0]["passed"] is True
        assert data[0]["score"] == pytest.approx(0.90)

    def test_async_eval_case_fails_below_threshold(self, tmp_path: Path):
        """An async @eval_case that returns score < threshold must exit 1."""
        from synapsekit.cli.test import run_test

        (tmp_path / "eval_async_fail.py").write_text(textwrap.dedent("""
            from synapsekit.evaluation.decorators import eval_case

            @eval_case(min_score=0.90)
            async def eval_failing_async():
                return {"score": 0.50}
        """))

        with pytest.raises(SystemExit) as exc_info:
            run_test(self._make_args(str(tmp_path)))
        assert exc_info.value.code == 1

    def test_async_eval_case_result_is_not_coroutine(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """The score in the output must be a float, never a coroutine object.

        This is the exact regression test for the v1.5.1 bug:
          TypeError: float() argument must be a string or a real number, not 'coroutine'
        """
        from synapsekit.cli.test import run_test

        (tmp_path / "eval_score_type.py").write_text(textwrap.dedent("""
            from synapsekit.evaluation.decorators import eval_case

            @eval_case(min_score=0.50)
            async def eval_score_is_float():
                return {"score": 0.75}
        """))

        run_test(self._make_args(str(tmp_path)))
        data = json.loads(capsys.readouterr().out)
        score = data[0]["score"]
        assert isinstance(score, float), (
            f"score must be float, got {type(score).__name__}. "
            "This likely means the async function was not awaited."
        )

    def test_mixed_sync_and_async_eval_cases(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """A file with both sync and async eval cases must run both correctly."""
        from synapsekit.cli.test import run_test

        (tmp_path / "eval_mixed.py").write_text(textwrap.dedent("""
            from synapsekit.evaluation.decorators import eval_case

            @eval_case(min_score=0.70)
            def eval_sync_case():
                return {"score": 0.80}

            @eval_case(min_score=0.70)
            async def eval_async_case():
                return {"score": 0.85, "cost_usd": 0.002}
        """))

        run_test(self._make_args(str(tmp_path)))
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 2
        assert all(r["passed"] for r in data)
        names = {r["name"] for r in data}
        assert "eval_sync_case" in names
        assert "eval_async_case" in names

    def test_async_eval_case_cost_and_latency_tracked(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """Cost and latency from async eval cases must appear in output."""
        from synapsekit.cli.test import run_test

        (tmp_path / "eval_meta.py").write_text(textwrap.dedent("""
            from synapsekit.evaluation.decorators import eval_case

            @eval_case(min_score=0.50, max_cost_usd=0.05, max_latency_ms=5000)
            async def eval_with_cost():
                return {"score": 0.80, "cost_usd": 0.005, "latency_ms": 200.0}
        """))

        run_test(self._make_args(str(tmp_path)))
        data = json.loads(capsys.readouterr().out)
        assert data[0]["passed"] is True
        assert data[0]["cost_usd"] == pytest.approx(0.005)
        assert data[0]["latency_ms"] == pytest.approx(200.0)

    def test_async_eval_case_exception_recorded(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        """An async eval case that raises must be recorded as failed with error."""
        from synapsekit.cli.test import run_test

        (tmp_path / "eval_exc.py").write_text(textwrap.dedent("""
            from synapsekit.evaluation.decorators import eval_case

            @eval_case(min_score=0.50)
            async def eval_that_raises():
                raise RuntimeError("LLM API error")
        """))

        with pytest.raises(SystemExit):
            run_test(self._make_args(str(tmp_path)))
        data = json.loads(capsys.readouterr().out)
        assert data[0]["passed"] is False
        assert "LLM API error" in data[0].get("failures", [""])[0] or data[0].get("score") == 0.0
