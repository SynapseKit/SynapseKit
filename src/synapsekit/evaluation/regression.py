"""EvalRegression — snapshot-based evaluation regression detection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EvalSnapshot:
    """Serialised evaluation results snapshot."""

    name: str
    timestamp: str
    results: list[dict[str, Any]]


@dataclass
class MetricDelta:
    """Change in a single metric between baseline and current."""

    case_name: str
    metric: str
    baseline: float
    current: float
    delta: float
    regressed: bool


@dataclass
class RegressionReport:
    """Comparison report between two evaluation snapshots."""

    baseline_name: str
    current_name: str
    deltas: list[MetricDelta] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return any(d.regressed for d in self.deltas)


# Default thresholds: 2% score drop, 10% cost increase, 20% latency increase.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "score": -0.02,
    "cost_usd": 0.10,
    "latency_ms": 0.20,
}


class EvalRegression:
    """Snapshot-based evaluation regression detection.

    Saves evaluation results as JSON snapshots and compares them to detect
    regressions in score, cost, or latency.

    Example::

        reg = EvalRegression(store_dir=".synapsekit_evals")
        reg.save_snapshot("v1.2", results)
        reg.save_snapshot("v1.3", new_results)
        report = reg.compare("v1.2", "v1.3")
        if report.has_regressions:
            print("Regressions detected!")
    """

    def __init__(self, store_dir: str = ".synapsekit_evals") -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, name: str, results: list[dict[str, Any]]) -> Path:
        """Save evaluation results as a named snapshot."""
        snapshot = EvalSnapshot(
            name=name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            results=results,
        )
        path = self._store_dir / f"{name}.json"
        path.write_text(json.dumps(asdict(snapshot), indent=2, default=str))
        return path

    def load_snapshot(self, name: str) -> EvalSnapshot:
        """Load a previously saved snapshot by name."""
        path = self._store_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")
        data = json.loads(path.read_text())
        return EvalSnapshot(**data)

    def list_snapshots(self) -> list[str]:
        """List all available snapshot names."""
        return sorted(p.stem for p in self._store_dir.glob("*.json"))

    def compare(
        self,
        baseline: str,
        current: str,
        thresholds: dict[str, float] | None = None,
    ) -> RegressionReport:
        """Compare two snapshots and report regressions."""
        thresh = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        baseline_snap = self.load_snapshot(baseline)
        current_snap = self.load_snapshot(current)

        # Index baseline results by case name
        baseline_by_name: dict[str, dict[str, Any]] = {
            r["name"]: r for r in baseline_snap.results if "name" in r
        }

        deltas: list[MetricDelta] = []
        for result in current_snap.results:
            case_name = result.get("name", "")
            baseline_result = baseline_by_name.get(case_name)
            if baseline_result is None:
                continue

            for metric in ("score", "cost_usd", "latency_ms"):
                b_val = baseline_result.get(metric)
                c_val = result.get(metric)
                if b_val is None or c_val is None:
                    continue

                delta = c_val - b_val
                # For score: regression = decrease beyond threshold (threshold is negative)
                # For cost/latency: regression = increase beyond threshold (threshold is positive %)
                if metric == "score":
                    regressed = delta < thresh.get("score", -0.02)
                else:
                    # Relative change
                    if b_val > 0:
                        relative = delta / b_val
                    else:
                        relative = 0.0
                    regressed = relative > thresh.get(metric, 0.10)

                deltas.append(
                    MetricDelta(
                        case_name=case_name,
                        metric=metric,
                        baseline=b_val,
                        current=c_val,
                        delta=delta,
                        regressed=regressed,
                    )
                )

        return RegressionReport(
            baseline_name=baseline,
            current_name=current,
            deltas=deltas,
        )
