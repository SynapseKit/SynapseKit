"""Regression tests for bounded output and non-hanging pipelines (#929)."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from synapsekit.shell import DirectShellExecutor, ShellCommand, ShellContext, ShellKind

# A cross-platform, unbounded stdout producer. Emits short newline-terminated
# lines forever so a downstream reader (and the output cap) has something to bite
# on regardless of pipe buffering.
_PRODUCER = "import sys\nwhile True:\n sys.stdout.write('x' * 63 + '\\n')\n sys.stdout.flush()"
_READ_ONE_LINE = "import sys; sys.stdout.write(sys.stdin.readline())"


def _context(tmp_path: Path) -> ShellContext:
    return ShellContext(cwd=str(tmp_path), platform=sys.platform, shell=ShellKind.BASH)


def _cmd(argv: tuple[str, ...], connector: str = "") -> ShellCommand:
    return ShellCommand(argv, " ".join(argv), connector)  # type: ignore[arg-type]


def test_unbounded_output_is_capped_and_does_not_hang(tmp_path: Path) -> None:
    executor = DirectShellExecutor(timeout=15.0, max_output_bytes=2000)
    producer = _cmd((sys.executable, "-c", _PRODUCER))

    started = time.monotonic()
    results = asyncio.run(executor.run_commands([producer], _context(tmp_path)))
    elapsed = time.monotonic() - started

    assert elapsed < 12.0  # capped and killed, nowhere near the 15s timeout
    assert len(results) == 1
    result = results[0]
    assert "[output truncated]" in result.stdout
    # Output is bounded by the cap (+ the short truncation marker), not unbounded.
    assert len(result.stdout.encode()) <= 2000 + len("\n[output truncated]")
    assert not result.timed_out


def test_pipeline_with_unbounded_producer_terminates(tmp_path: Path) -> None:
    executor = DirectShellExecutor(timeout=15.0, max_output_bytes=2000)
    commands = [
        _cmd((sys.executable, "-c", _PRODUCER)),
        _cmd((sys.executable, "-c", _READ_ONE_LINE), connector="|"),
    ]

    started = time.monotonic()
    results = asyncio.run(executor.run_commands(commands, _context(tmp_path)))
    elapsed = time.monotonic() - started

    assert elapsed < 12.0  # the classic `yes | head -1` hang is gone
    assert len(results) == 2
    # The consumer ran on the producer's (truncated) output and exited cleanly.
    consumer = results[-1]
    assert consumer.exit_code == 0
    assert consumer.stdout.startswith("x")


def test_bounded_output_below_cap_is_untouched(tmp_path: Path) -> None:
    executor = DirectShellExecutor(timeout=10.0, max_output_bytes=1_000_000)
    echo = _cmd((sys.executable, "-c", "print('hello world')"))

    results = asyncio.run(executor.run_commands([echo], _context(tmp_path)))

    assert results[0].exit_code == 0
    assert results[0].stdout.strip() == "hello world"
    assert "[output truncated]" not in results[0].stdout
