"""Agent benchmarking suites.

This subpackage integrates industry-standard agent benchmarks like GAIA,
SWE-bench, WebArena, and AgentBench into SynapseKit as first-class evaluation suites.
"""

from __future__ import annotations

from .agentbench import AgentBenchBenchmark
from .base import BaseBenchmark, BenchmarkResult
from .gaia import GAIABenchmark
from .swe_bench import SWEBenchmark
from .webarena import WebArenaBenchmark

BENCHMARK_REGISTRY: dict[str, type[BaseBenchmark]] = {
    "gaia": GAIABenchmark,
    "swe-bench": SWEBenchmark,
    "webarena": WebArenaBenchmark,
    "agentbench": AgentBenchBenchmark,
}

__all__ = [
    "AgentBenchBenchmark",
    "BaseBenchmark",
    "BenchmarkResult",
    "GAIABenchmark",
    "SWEBenchmark",
    "WebArenaBenchmark",
    "BENCHMARK_REGISTRY",
]
