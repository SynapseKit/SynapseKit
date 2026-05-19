from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    qs = statistics.quantiles(samples, n=100, method="inclusive")
    return {
        "p50": qs[49],
        "p95": qs[94],
        "p99": qs[98],
    }


def _extract_samples(bench: dict) -> list[float]:
    stats = bench.get("stats", {})
    for key in ("data", "samples", "values", "raw"):
        if isinstance(stats.get(key), list):
            return stats[key]
    return []


def main(path: str) -> int:
    json_path = Path(path)
    if not json_path.exists():
        fallback = Path(__file__).resolve().parents[1] / "benchmark.json"
        if fallback.exists():
            json_path = fallback
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    benches = payload.get("benchmarks", [])
    for bench in benches:
        name = bench.get("name", "unknown")
        samples = _extract_samples(bench)
        pcts = _percentiles(samples)
        print(
            f"{name}: p50={pcts['p50']:.6f}  p95={pcts['p95']:.6f}  p99={pcts['p99']:.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
