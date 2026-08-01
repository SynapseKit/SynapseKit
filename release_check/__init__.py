"""Reusable release-validation harness for SynapseKit.

Run real, no-mock checks that a release build actually works, and emit a
single pass/skip/fail report. Reusable every release via `make release-check`
(offline) or the `release-validation` scheduled workflow (offline + live).

Layers (each is a self-contained check that reports pass/skip/fail):

  1. core-import      — `import synapsekit` + CLI entrypoint in a fresh
                        interpreter with every optional dependency blocked, so a
                        bare `pip install synapsekit` (no extras) is proven
                        importable. Folds in the check the core Docker image runs.
  2. export-surface   — every name in `synapsekit.__all__` resolves (no broken
                        lazy-import wiring / typo'd exports).
  3. functional       — the real functional smoke test (`smoke_test.py`):
                        loaders, splitters, graph, tools, embeddings, vector
                        stores, and (in --live mode) real LLM completions.

See `python -m release_check --help`.
"""
