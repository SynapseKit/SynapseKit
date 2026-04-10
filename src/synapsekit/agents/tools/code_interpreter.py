from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from ..base import BaseTool, ToolResult

_WORKER_SCRIPT = textwrap.dedent(
    r"""
    from __future__ import annotations

    import contextlib
    import io
    import json
    import os
    import sys
    import traceback
    from pathlib import Path


    def file_entry(path: Path, workspace: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(workspace).as_posix(),
            "size": path.stat().st_size,
        }


    def apply_memory_limit(memory_limit_mb: int) -> None:
        if memory_limit_mb <= 0:
            return
        with contextlib.suppress(ImportError, OSError, ValueError):
            import resource

            limit_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))


    def capture_plots(workspace: Path) -> list[dict[str, object]]:
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is None:
            return []
        get_fignums = getattr(plt, "get_fignums", None)
        figure = getattr(plt, "figure", None)
        if not callable(get_fignums) or not callable(figure):
            return []

        plot_dir = workspace / "plots"
        plot_dir.mkdir(exist_ok=True)

        plots = []
        for idx, fig_num in enumerate(get_fignums(), start=1):
            fig = figure(fig_num)
            savefig = getattr(fig, "savefig", None)
            if not callable(savefig):
                continue
            plot_path = plot_dir / f"plot_{idx}.png"
            savefig(plot_path)
            plots.append(file_entry(plot_path, workspace))
        return plots


    def collect_dataframes(namespace: dict[str, object]) -> list[dict[str, str]]:
        dataframes = []
        for name, value in namespace.items():
            if name.startswith("__"):
                continue
            if type(value).__module__.startswith("pandas"):
                dataframes.append({"name": name, "repr": repr(value)})
        return dataframes


    def list_files(workspace: Path) -> list[dict[str, object]]:
        files = [
            file_entry(path, workspace)
            for path in workspace.rglob("*")
            if path.is_file() and path.name != "_code_interpreter_worker.py"
        ]
        return sorted(files, key=lambda item: item["path"])


    def main() -> None:
        workspace = Path(sys.argv[1]).resolve()
        memory_limit_mb = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        code = sys.stdin.read()

        workspace.mkdir(parents=True, exist_ok=True)
        os.chdir(workspace)
        apply_memory_limit(memory_limit_mb)

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        namespace: dict[str, object] = {"__name__": "__main__"}
        payload: dict[str, object] = {
            "stdout": "",
            "stderr": "",
            "files": [],
            "plots": [],
            "dataframes": [],
        }
        error = None

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(code, namespace, namespace)
        except MemoryError:
            error = "Memory limit exceeded during code execution"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=stderr_buffer)
        finally:
            payload["stdout"] = stdout_buffer.getvalue()
            payload["stderr"] = stderr_buffer.getvalue()
            payload["plots"] = capture_plots(workspace)
            payload["dataframes"] = collect_dataframes(namespace)
            payload["files"] = list_files(workspace)

        if error is not None:
            payload["error"] = error

        sys.stdout.write(json.dumps(payload))


    if __name__ == "__main__":
        main()
    """
)


class CodeInterpreterTool(BaseTool):
    """Execute Python in an isolated subprocess and capture outputs/artifacts."""

    name = "code_interpreter"
    description = (
        "Execute Python code in an isolated subprocess and capture stdout, stderr, "
        "generated files, plots, and dataframe reprs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in the sandboxed interpreter",
            }
        },
        "required": ["code"],
    }

    def __init__(self, timeout: float = 5.0, memory_limit_mb: int = 256) -> None:
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb

    async def run(self, code: str = "", **kwargs: Any) -> ToolResult:
        src = code or kwargs.get("input", "")
        if not src:
            return ToolResult(output="", error="No code provided.")
        return await asyncio.to_thread(self._run_sync, src)

    def _run_sync(self, code: str) -> ToolResult:
        with tempfile.TemporaryDirectory(prefix="synapsekit-code-interpreter-") as tmpdir:
            workspace = Path(tmpdir)
            worker_path = workspace / "_code_interpreter_worker.py"
            worker_path.write_text(_WORKER_SCRIPT, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(worker_path), str(workspace), str(self.memory_limit_mb)],
                    input=code,
                    capture_output=True,
                    text=True,
                    cwd=workspace,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    output="", error=f"Code execution timed out after {self.timeout} seconds"
                )

            raw_stdout = proc.stdout.strip()
            raw_stderr = proc.stderr.strip()

            if not raw_stdout:
                error = raw_stderr or "Code execution failed unexpectedly"
                if proc.returncode != 0 and not raw_stderr:
                    error = "Memory limit exceeded during code execution"
                return ToolResult(output="", error=error)

            try:
                payload = json.loads(raw_stdout)
            except json.JSONDecodeError:
                error = raw_stderr or raw_stdout or "Code execution failed unexpectedly"
                return ToolResult(output="", error=error)

            error = payload.pop("error", None)
            if proc.returncode != 0 and error is None:
                error = raw_stderr or "Memory limit exceeded during code execution"

            return ToolResult(output=json.dumps(payload), error=error)
