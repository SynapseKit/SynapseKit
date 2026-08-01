"""A bare `pip install synapsekit` (no extras) must be importable.

Optional third-party libraries (httpx, boto3, chromadb, ...) belong to extras and
are NOT installed in a core-only environment. If any module in the import graph of
`import synapsekit` imports one at module top level, the whole package becomes
unimportable on a minimal install — which is exactly what shipped in the core
Docker image before this guard existed (notion.py did a top-level `import httpx`).

These tests run a *fresh* interpreter with every optional dependency blocked, so
they fail on a regression regardless of what the dev environment happens to have
installed. Keep this list in sync with the optional-dependency import names as new
extras are added.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Import names (top-level module names) of optional dependencies — everything
# outside the core deps (numpy, rank-bm25, pillow, pyasn1, cryptography).
_OPTIONAL_IMPORT_NAMES = [
    "httpx",
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

_BLOCK_AND_IMPORT = textwrap.dedent(
    """
    import importlib.abc, sys
    blocked = set({blocked!r})

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in blocked:
                raise ModuleNotFoundError("No module named '%s' (blocked)" % name)
            return None

    sys.meta_path.insert(0, _Blocker())
    for _m in [k for k in list(sys.modules) if k.split(".")[0] in blocked]:
        del sys.modules[_m]

    import synapsekit
    from synapsekit.cli.main import main  # the console-script entrypoint
    assert synapsekit.__version__
    print("core-import-ok", synapsekit.__version__)
    """
)


def _run_core_import(blocked: list[str]) -> subprocess.CompletedProcess[str]:
    script = _BLOCK_AND_IMPORT.format(blocked=blocked)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )


def test_import_synapsekit_without_any_optional_dep() -> None:
    """`import synapsekit` + the CLI entrypoint work with all extras absent."""
    result = _run_core_import(_OPTIONAL_IMPORT_NAMES)
    assert result.returncode == 0, (
        f"bare `import synapsekit` failed with optional deps blocked:\n{result.stderr}"
    )
    assert "core-import-ok" in result.stdout


def test_import_synapsekit_without_httpx() -> None:
    """Regression for the notion.py top-level `import httpx` (broke the core image)."""
    result = _run_core_import(["httpx"])
    assert result.returncode == 0, (
        f"bare `import synapsekit` failed with httpx blocked:\n{result.stderr}"
    )
