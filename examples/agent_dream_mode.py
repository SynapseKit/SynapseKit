"""Minimal explicit Dream Mode example.

Run with a configured local ``BaseLLM``/``EdgeRuntime`` in place of the
``backend`` placeholder. No background work begins until ``run_once`` or
``run_forever`` is called.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from synapsekit.audit import AuditTracer, EventKind
from synapsekit.dream import DreamConfig, DreamMode, PowerStatus


async def main() -> None:
    tracer = AuditTracer(run_id="example-day")
    tracer.record(EventKind.ERROR, {"message": "retrieval failed; should retry with mesh"})
    dream = DreamMode(
        config=DreamConfig(
            schedule="idle_30m or 02:00",
            state_path=Path(".synapsekit/dream/state.sqlite3"),
            audit_dir=Path(".synapsekit/dream/audit"),
        ),
        memory_paths=["MEMORY.md"],
    )
    await dream.ingest_traces(tracer.records)
    report = await dream.run_once(
        force=True,
        power=PowerStatus(plugged_in=True, battery_percent=100),
    )
    print(dream.morning_briefing(report))
    dream.close()


if __name__ == "__main__":
    asyncio.run(main())
