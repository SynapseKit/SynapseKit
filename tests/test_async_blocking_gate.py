"""Tests for the async-blocking static gate (scripts/check_async_blocking.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_async_blocking",
    Path(__file__).resolve().parent.parent / "scripts" / "check_async_blocking.py",
)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


def _check(src: str, tmp_path: Path) -> list[tuple[int, str]]:
    f = tmp_path / "sample.py"
    f.write_text(src, encoding="utf-8")
    return checker.check_file(f)


def test_flags_blocking_write_in_async(tmp_path):
    src = "from pathlib import Path\nasync def save(p, data):\n    Path(p).write_text(data)\n"
    violations = _check(src, tmp_path)
    assert len(violations) == 1
    assert ".write_text()" in violations[0][1]
    assert violations[0][0] == 3


def test_flags_subprocess_and_sleep_and_open(tmp_path):
    src = (
        "import subprocess, time\n"
        "async def go():\n"
        "    subprocess.run(['ls'])\n"
        "    time.sleep(1)\n"
        "    open('x')\n"
    )
    violations = _check(src, tmp_path)
    labels = " ".join(m for _, m in violations)
    assert "subprocess.run()" in labels
    assert "time.sleep()" in labels
    assert "open()" in labels
    assert len(violations) == 3


def test_offloaded_nested_helper_is_clean(tmp_path):
    src = (
        "import asyncio\n"
        "from pathlib import Path\n"
        "async def save(p, data):\n"
        "    def _write():\n"
        "        Path(p).write_text(data)\n"
        "    await asyncio.to_thread(_write)\n"
    )
    assert _check(src, tmp_path) == []


def test_offloaded_callable_reference_is_clean(tmp_path):
    src = (
        "import asyncio\n"
        "from pathlib import Path\n"
        "async def load(p):\n"
        "    return await asyncio.to_thread(Path(p).read_text)\n"
    )
    assert _check(src, tmp_path) == []


def test_awaited_call_is_not_flagged(tmp_path):
    # An async client whose read_bytes() is a coroutine -- awaited, so fine.
    src = "async def load(client):\n    return await client.read_bytes()\n"
    assert _check(src, tmp_path) == []


def test_sync_function_is_ignored(tmp_path):
    src = "from pathlib import Path\ndef save(p, data):\n    Path(p).write_text(data)\n"
    assert _check(src, tmp_path) == []


def test_suppression_comment_is_respected(tmp_path):
    src = (
        "from pathlib import Path\n"
        "async def save(p, data):\n"
        "    Path(p).write_text(data)  # allow-blocking: tiny sentinel file\n"
    )
    assert _check(src, tmp_path) == []


def test_real_source_tree_is_clean(tmp_path):
    # The gate must stay green on the actual package.
    src_root = Path(__file__).resolve().parent.parent / "src" / "synapsekit"
    total = 0
    for path in src_root.rglob("*.py"):
        total += len(checker.check_file(path))
    assert total == 0, "async-blocking violations present in src/synapsekit"
