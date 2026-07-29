#!/usr/bin/env python3
"""Static gate: no blocking IO directly inside an ``async def`` body.

SynapseKit is async-first. A blocking file/subprocess/sleep/network call executed
directly in a coroutine freezes the event loop. The house rule is to offload such
work via ``asyncio.to_thread(...)`` or ``loop.run_in_executor(...)`` -- i.e. move
the blocking statement into a nested sync helper (or a bare callable reference) and
``await`` the offload.

This checker walks every ``async def`` and flags blocking calls that sit *directly*
in its own execution frame. It deliberately does NOT descend into nested ``def`` /
``async def`` / ``lambda`` bodies, because that is precisely where offloaded work
lives::

    async def save(self):                     # OK
        def _write():
            self.path.write_text(data)         # nested sync helper -- not flagged
        await asyncio.to_thread(_write)

    async def load(self):                     # OK
        return await asyncio.to_thread(self.path.read_text)  # reference, not a call

    async def bad(self):                      # FLAGGED
        self.path.write_text(data)             # blocks the event loop

A genuinely-async method that happens to share a blocked name (e.g. an async client
whose ``.read_bytes()`` is a coroutine) is awaited, so awaited calls are never
flagged. For a deliberate, reviewed exception, add a trailing ``# allow-blocking``
comment on the offending line.

Exit code 0 if clean, 1 if any violation is found. Scans ``src/synapsekit`` by
default, or the paths passed as arguments.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# obj.<name>(...) -- pathlib / common blocking filesystem methods
BLOCKING_METHODS = {
    "read_text",
    "write_text",
    "read_bytes",
    "write_bytes",
    "mkdir",
}
# <module>.<name>(...) -- blocking stdlib / network calls
BLOCKING_DOTTED = {
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("subprocess", "Popen"),
    ("os", "system"),
    ("time", "sleep"),
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("requests", "delete"),
    ("requests", "patch"),
    ("requests", "head"),
    ("requests", "request"),
}
# bare builtin calls
BLOCKING_BUILTINS = {"open"}

SUPPRESS = "allow-blocking"


def _same_frame_nodes(node: ast.AST):
    """Yield descendants that execute in ``node``'s own frame.

    Does not cross into nested function/lambda definitions -- offloaded blocking
    work is expected to live inside those.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _same_frame_nodes(child)


def _describe(call: ast.Call) -> str | None:
    """Return a human label if ``call`` is a blocking call, else None."""
    func = call.func
    if isinstance(func, ast.Attribute):
        attr = func.attr
        mod = func.value.id if isinstance(func.value, ast.Name) else None
        if attr in BLOCKING_METHODS:
            return f".{attr}()"
        if mod is not None and (mod, attr) in BLOCKING_DOTTED:
            return f"{mod}.{attr}()"
    elif isinstance(func, ast.Name) and func.id in BLOCKING_BUILTINS:
        return f"{func.id}()"
    return None


def check_file(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - surfaced as a hard error
        return [(exc.lineno or 0, f"could not parse: {exc.msg}")]

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        awaited = {id(n.value) for n in _same_frame_nodes(node) if isinstance(n, ast.Await)}
        for inner in _same_frame_nodes(node):
            if not isinstance(inner, ast.Call) or id(inner) in awaited:
                continue
            label = _describe(inner)
            if label is None:
                continue
            line_text = lines[inner.lineno - 1] if inner.lineno <= len(lines) else ""
            if SUPPRESS in line_text:
                continue
            violations.append(
                (
                    inner.lineno,
                    f"blocking `{label}` in async `{node.name}` "
                    f"-- offload via asyncio.to_thread / run_in_executor "
                    f"(or add `# {SUPPRESS}` if intentional)",
                )
            )
    return violations


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv] or [Path("src/synapsekit")]
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.suffix == ".py":
            files.append(target)

    total = 0
    for path in files:
        for lineno, msg in check_file(path):
            print(f"{path}:{lineno}: {msg}")
            total += 1

    if total:
        print(f"\nasync-blocking gate: {total} violation(s) found.", file=sys.stderr)
        return 1
    print(f"async-blocking gate: clean ({len(files)} files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
