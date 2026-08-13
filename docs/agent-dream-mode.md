# Dream Mode

Dream Mode is an explicit, local-first overnight reflection runner. It reads
completed `AuditTracer` records, validates their hash chains, distills durable
corrections, proposes (but does not apply) `LivingMemory` patches, refreshes a
`KnowledgeMesh`, flags stale memory files, and writes a signed audit bundle.

Nothing starts in the background merely because the module is imported. A
caller opts into a one-shot run or the cancellable scheduler:

```python
from synapsekit.dream import DreamConfig, DreamMode, PowerStatus

dream = DreamMode(
    config=DreamConfig(schedule="idle_30m or 02:00", budget_tokens=100_000),
    memory_paths=["CLAUDE.md", "MEMORY.md"],
    backend=edge_runtime,  # EdgeRuntime keeps inference local by default
)
dream.ingest_traces(tracer.records)
report = await dream.run_once(
    force=True,
    power=PowerStatus(plugged_in=True, battery_percent=100),
)
print(dream.morning_briefing(report))
dream.close()
```

The same flow is available from the CLI:

```text
synapsekit dream run --force --trace-bundle ./day.audit.zip --memory-path ./MEMORY.md --json
synapsekit dream status
```

Safety boundaries:

- Trace input is local and replay-checked; malformed or broken chains are not
  distilled.
- Cloud fallback remains governed by the supplied `EdgeRuntime` policy.
- Memory output is a signed, pending `LivingMemory` patch and requires normal
  human review before application.
- Entity consolidation reports duplicate candidates; it never merges or
  deletes files automatically.
- Stale-memory pruning is a report-only operation.
- The default scheduler fails closed when power state is unknown or the
  machine is not plugged in.

The state database and audit bundles default to `~/.synapsekit/dream/` and can
be redirected with `DreamConfig` or the CLI flags. The default tasks are
`distill_lessons`, `propose_memory_patches`, `consolidate_entities`, and
`prune_stale`.
