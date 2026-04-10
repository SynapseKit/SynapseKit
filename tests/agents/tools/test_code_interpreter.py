from __future__ import annotations

import json

import pytest

from synapsekit.agents.tools import CodeInterpreterTool


def parse_output(raw: str) -> dict:
    return json.loads(raw)


@pytest.mark.asyncio
async def test_code_interpreter_captures_stdout_stderr_and_files(tmp_path) -> None:
    tool = CodeInterpreterTool(timeout=5.0, memory_limit_mb=256)
    code = (
        "import pathlib, sys\n"
        "print('hello stdout')\n"
        "print('hello stderr', file=sys.stderr)\n"
        "pathlib.Path('artifact.txt').write_text('artifact data', encoding='utf-8')\n"
    )

    res = await tool.run(code=code)

    assert not res.is_error
    payload = parse_output(res.output)
    assert payload["stdout"].strip() == "hello stdout"
    assert payload["stderr"].strip() == "hello stderr"
    assert payload["files"][0]["path"] == "artifact.txt"


@pytest.mark.asyncio
async def test_code_interpreter_captures_fake_matplotlib_plots() -> None:
    tool = CodeInterpreterTool(timeout=5.0, memory_limit_mb=256)
    code = """
import sys, types

class FakeFigure:
    def __init__(self, number):
        self.number = number
    def savefig(self, path):
        with open(path, 'wb') as fh:
            fh.write(f'figure-{self.number}'.encode('utf-8'))

plt = types.ModuleType('matplotlib.pyplot')
plt._figures = {1: FakeFigure(1)}
plt.get_fignums = lambda: list(plt._figures)
plt.figure = lambda number: plt._figures[number]
matplotlib = types.ModuleType('matplotlib')
matplotlib.pyplot = plt
sys.modules['matplotlib'] = matplotlib
sys.modules['matplotlib.pyplot'] = plt
"""

    res = await tool.run(code=code)

    assert not res.is_error
    payload = parse_output(res.output)
    assert len(payload["plots"]) == 1
    assert payload["plots"][0]["path"].endswith(".png")


@pytest.mark.asyncio
async def test_code_interpreter_captures_fake_pandas_dataframes() -> None:
    tool = CodeInterpreterTool(timeout=5.0, memory_limit_mb=256)
    code = """
class FakeDataFrame:
    __module__ = 'pandas.core.frame'
    def __repr__(self):
        return 'col\\n1'

df = FakeDataFrame()
"""

    res = await tool.run(code=code)

    assert not res.is_error
    payload = parse_output(res.output)
    assert payload["dataframes"] == [{"name": "df", "repr": "col\n1"}]


@pytest.mark.asyncio
async def test_code_interpreter_times_out() -> None:
    tool = CodeInterpreterTool(timeout=1.0, memory_limit_mb=256)

    res = await tool.run(code="while True:\n    pass")

    assert res.is_error
    assert "timed out" in (res.error or "").lower()


@pytest.mark.asyncio
async def test_code_interpreter_enforces_memory_limit() -> None:
    tool = CodeInterpreterTool(timeout=5.0, memory_limit_mb=64)

    res = await tool.run(code="x = bytearray(1024 * 1024 * 256)\nprint(len(x))")

    assert res.is_error
    assert "memory" in (res.error or "").lower()


@pytest.mark.asyncio
async def test_top_level_export() -> None:
    from synapsekit import CodeInterpreterTool as TopCodeInterpreterTool

    assert TopCodeInterpreterTool is CodeInterpreterTool
