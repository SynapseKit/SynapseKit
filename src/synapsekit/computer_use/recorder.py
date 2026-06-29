from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import ComputerAction, ComputerObservation, ComputerStep, SafetyCheck


@dataclass
class RecordedSession:
    """Replayable session loaded from a JSONL recording."""

    path: Path
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def observations(self) -> list[ComputerObservation]:
        result: list[ComputerObservation] = []
        for event in self.events:
            if event.get("event") == "observation":
                result.append(_observation_from_record(event["observation"]))
        return result

    @property
    def actions(self) -> list[ComputerAction]:
        result: list[ComputerAction] = []
        for event in self.events:
            if event.get("event") == "action":
                result.append(ComputerAction(**event["action"]))
        return result


class SessionRecorder:
    """Write computer-use runs as newline-delimited JSON events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False

    def start(self, task: str) -> None:
        self._write({"event": "start", "task": task})

    def record_observation(self, observation: ComputerObservation) -> None:
        self._write({"event": "observation", "observation": _observation_to_record(observation)})

    def record_action(self, action: ComputerAction, safety: SafetyCheck) -> None:
        self._write(
            {
                "event": "action",
                "action": _safe_asdict(action),
                "safety": _safe_asdict(safety),
            }
        )

    def record_step(self, step: ComputerStep) -> None:
        self._write(
            {
                "event": "step",
                "step": {
                    "observation": _observation_to_record(step.observation),
                    "action": _safe_asdict(step.action),
                    "safety": _safe_asdict(step.safety),
                    "output": step.output,
                    "error": step.error,
                },
            }
        )

    def finish(self, *, completed: bool, output: str = "", error: str | None = None) -> None:
        self._write({"event": "finish", "completed": completed, "output": output, "error": error})
        self._closed = True

    @classmethod
    def load(cls, path: str | Path) -> RecordedSession:
        target = Path(path)
        events: list[dict[str, Any]] = []
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
        return RecordedSession(path=target, events=events)

    def _write(self, event: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed SessionRecorder.")
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")


def _observation_to_record(observation: ComputerObservation) -> dict[str, Any]:
    data = _safe_asdict(observation)
    screenshot = observation.screenshot
    data["screenshot"] = base64.b64encode(screenshot).decode("ascii") if screenshot else None
    data["screenshot_encoding"] = "base64" if screenshot else None
    return data


def _observation_from_record(data: dict[str, Any]) -> ComputerObservation:
    screenshot_value = data.get("screenshot")
    payload = dict(data)
    payload.pop("screenshot_encoding", None)
    if screenshot_value:
        payload["screenshot"] = base64.b64decode(screenshot_value)
    return ComputerObservation(**payload)


def _safe_asdict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
