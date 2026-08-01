"""Release-validation harness entrypoint. See release_check/__init__.py.

Usage:
    python -m release_check                 # offline layers, human-readable
    python -m release_check --live          # also run live LLM checks (needs API keys)
    python -m release_check --json out.json --md out.md
    python -m release_check --only core-import export-surface

Exit code is non-zero if any check FAILS (SKIP never fails the run).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASS, FAIL, SKIP = "pass", "fail", "skip"

# Import names of optional dependencies — everything outside the core deps
# (numpy, rank-bm25, pillow, pyasn1, cryptography). A bare install must not need
# any of these to `import synapsekit`. Mirrors tests/test_core_import_without_extras.py.
OPTIONAL_IMPORT_NAMES = [
    "httpx",
    "yaml",
    "bs4",
    "lxml",
    "chromadb",
    "faiss",
    "qdrant_client",
    "pinecone",
    "weaviate",
    "pymilvus",
    "lancedb",
    "sqlite_vec",
    "kuzu",
    "pgvector",
    "ollama",
    "ai21",
    "cohere",
    "mistralai",
    "boto3",
    "duckduckgo_search",
    "huggingface_hub",
    "serpapi",
    "tavily",
    "mcp",
    "redis",
    "aiomcache",
    "fastapi",
    "starlette",
    "uvicorn",
    "psycopg",
    "asyncpg",
    "groq",
    "croniter",
    "erniebot",
    "wolframalpha",
    "discord",
    "googleapiclient",
    "feedparser",
    "git",
    "hubspot",
    "supabase",
    "pymongo",
    "simple_salesforce",
    "azure",
    "elasticsearch",
    "snowflake",
    "pyairtable",
    "playwright",
    "pyautogui",
    "astrapy",
    "clickhouse_connect",
    "duckdb",
    "networkx",
    "neo4j",
    "openai",
    "anthropic",
    "arxiv",
    "replicate",
    "requests",
    "aiohttp",
    "marqo",
    "opensearchpy",
    "cassandra",
    "youtube_transcript_api",
]

# LLM providers whose API key, if present in the environment, is exercised by a
# real 1-token completion in the functional (live) layer via smoke_test.py.
LIVE_KEY_ENV_VARS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]


def _result(name: str, status: str, note: str = "") -> dict:
    return {"name": name, "status": status, "note": note}


def layer_core_import() -> list[dict]:
    """Import synapsekit + the CLI entrypoint with all optional deps blocked."""
    script = textwrap.dedent(
        f"""
        import importlib.abc, sys
        blocked = set({OPTIONAL_IMPORT_NAMES!r})

        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in blocked:
                    raise ModuleNotFoundError("No module named '%s' (blocked)" % name)
                return None

        sys.meta_path.insert(0, _Blocker())
        for _m in [k for k in list(sys.modules) if k.split(".")[0] in blocked]:
            del sys.modules[_m]

        import synapsekit
        from synapsekit.cli.main import main
        print(synapsekit.__version__)
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if proc.returncode == 0:
        return [_result("import synapsekit (no extras)", PASS, proc.stdout.strip())]
    last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "failed"
    return [_result("import synapsekit (no extras)", FAIL, last)]


def layer_export_surface() -> list[dict]:
    """Every name in synapsekit.__all__ must resolve."""
    import synapsekit

    resolved = missing = 0
    broken: list[dict] = []
    for name in synapsekit.__all__:
        try:
            getattr(synapsekit, name)
            resolved += 1
        except ModuleNotFoundError:
            missing += 1  # optional extra not installed here — fine
        except Exception as exc:  # AttributeError / import wiring bug
            broken.append(_result(f"export: {name}", FAIL, f"{type(exc).__name__}: {exc}"))

    summary = _result(
        f"__all__ exports resolve ({resolved} ok, {missing} skipped-missing-extra)",
        PASS if not broken else FAIL,
        f"{len(broken)} broken" if broken else "",
    )
    return [summary, *broken]


def layer_functional(live: bool) -> list[dict]:
    """Run the real functional smoke test; parse its pass/fail/skip summary."""
    smoke = ROOT / "smoke_test.py"
    if not smoke.exists():
        return [_result("functional smoke test", SKIP, "smoke_test.py not found")]

    import os

    env = dict(os.environ)
    if not live:
        # Offline: force live-LLM checks in smoke_test.py to skip (no network).
        for var in LIVE_KEY_ENV_VARS:
            env.pop(var, None)

    proc = subprocess.run(
        [sys.executable, str(smoke)], capture_output=True, text=True, env=env, cwd=str(ROOT)
    )
    m = re.search(r"(\d+)\s+passed\D+(\d+)\s+failed\D+(\d+)\s+skipped", proc.stdout)
    if not m:
        return [_result("functional smoke test", FAIL, "could not parse smoke_test.py output")]
    passed, failed, skipped = (int(m.group(i)) for i in (1, 2, 3))
    status = PASS if failed == 0 and proc.returncode == 0 else FAIL
    note = f"{passed} passed · {failed} failed · {skipped} skipped"
    return [_result("functional smoke test", status, note)]


LAYERS = {
    "core-import": layer_core_import,
    "export-surface": layer_export_surface,
    # functional takes `live`; wrapped below.
}


def run(only: list[str] | None, live: bool) -> dict:
    selected = only or ["core-import", "export-surface", "functional"]
    layers: dict[str, list[dict]] = {}
    for name in selected:
        if name == "functional":
            layers[name] = layer_functional(live)
        elif name in LAYERS:
            layers[name] = LAYERS[name]()
        else:
            layers[name] = [_result(name, FAIL, "unknown layer")]
    return layers


def render_markdown(layers: dict, live: bool) -> str:
    lines = ["# SynapseKit release-validation report", ""]
    lines.append(f"Mode: **{'live' if live else 'offline'}**")
    lines.append("")
    total = {PASS: 0, FAIL: 0, SKIP: 0}
    for layer, checks in layers.items():
        lines.append(f"## {layer}")
        lines.append("")
        lines.append("| status | check | note |")
        lines.append("|---|---|---|")
        for c in checks:
            total[c["status"]] += 1
            icon = {"pass": "✅", "fail": "❌", "skip": "⚠️"}[c["status"]]
            lines.append(f"| {icon} | {c['name']} | {c['note']} |")
        lines.append("")
    lines.append(f"**{total[PASS]} passed · {total[FAIL]} failed · {total[SKIP]} skipped**")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release_check", description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="also run live LLM checks (needs API keys)"
    )
    parser.add_argument("--only", nargs="+", metavar="LAYER", help="run only these layers")
    parser.add_argument("--json", dest="json_path", help="write JSON report to this path")
    parser.add_argument("--md", dest="md_path", help="write markdown report to this path")
    args = parser.parse_args(argv)

    layers = run(args.only, args.live)

    md = render_markdown(layers, args.live)
    print(md)

    failed = sum(1 for checks in layers.values() for c in checks if c["status"] == FAIL)

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps({"live": args.live, "failed": failed, "layers": layers}, indent=2)
        )
    if args.md_path:
        Path(args.md_path).write_text(md)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
