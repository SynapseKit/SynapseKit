"""Dream Mode orchestration: replay, distill, propose, consolidate, report."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from ..audit import (
    GENESIS_HASH,
    AuditRecord,
    AuditTracer,
    EventKind,
    PIIRedactor,
    ReplayEngine,
    SigningPolicy,
    export_audit_bundle,
    load_bundle,
)
from ..memory import LivingMemory
from ..mesh import KnowledgeMesh
from .distill import (
    DeterministicLessonDistiller,
    LessonBackend,
    ModelLessonDistiller,
    estimate_tokens,
    trace_transcript,
)
from .scheduler import (
    DreamSchedule,
    DreamScheduler,
    IdleMonitor,
    PowerMonitor,
    SystemIdleMonitor,
    SystemPowerMonitor,
    wait_for_stop,
)
from .store import DreamStateStore
from .types import (
    DreamConfig,
    DreamRunResult,
    Lesson,
    MeshConsolidation,
    PowerStatus,
    StaleMemory,
    TraceWindow,
)

logger = logging.getLogger(__name__)


class DreamMode:
    """Run bounded, local-first overnight reflection.

    Dream Mode is opt-in: constructing this class never starts a background
    task, changes memory files, or contacts a cloud provider. ``run_once``
    must be called explicitly, and LivingMemory keeps every generated patch in
    its normal pending-for-review state.
    """

    def __init__(
        self,
        *,
        config: DreamConfig | None = None,
        backend: LessonBackend | None = None,
        memory: LivingMemory | None = None,
        memory_paths: Sequence[str | Path] = (),
        mesh: KnowledgeMesh | None = None,
        state_store: DreamStateStore | None = None,
        power_monitor: PowerMonitor | None = None,
        idle_monitor: IdleMonitor | None = None,
        signing_policy: SigningPolicy | None = None,
    ) -> None:
        self.config = config or DreamConfig()
        self.backend = backend
        self.memory_paths = tuple(str(Path(path).expanduser()) for path in memory_paths)
        self.mesh = mesh
        self.state = state_store or DreamStateStore(self.config.state_path)
        self.power_monitor = power_monitor or SystemPowerMonitor()
        self.idle_monitor = idle_monitor or SystemIdleMonitor()
        self.signing_policy = signing_policy or SigningPolicy.ed25519()
        self.scheduler = DreamScheduler(
            DreamSchedule.parse(
                self.config.schedule,
                default_idle_seconds=self.config.idle_after_seconds,
            )
        )
        self._owns_state = state_store is None
        self._stop_event = asyncio.Event()

        self.memory: LivingMemory | None
        if memory is not None:
            self.memory = memory
        elif self.memory_paths:
            self.memory = LivingMemory(
                list(self.memory_paths),
                proposer=backend,
                require_approval=True,
                occurrence_threshold=1,
                store_path=str(
                    Path(self.config.state_path).expanduser().with_name("memory_patches.jsonl")
                ),
                occurrence_path=str(
                    Path(self.config.state_path).expanduser().with_name("memory_occurrences.json")
                ),
            )
        else:
            self.memory = None

    def close(self) -> None:
        """Close the local state database when the caller owns it."""

        if self._owns_state:
            self.state.close()

    def stop(self) -> None:
        """Request that a running scheduler stop after its current iteration."""

        self._stop_event.set()

    def ingest_traces(self, records: Iterable[AuditRecord]) -> int:
        """Persist completed local audit traces for a future dream run."""

        records_list = list(records)
        inserted = self.state.append_traces(records_list)
        self.state.update_memory_reads(records_list)
        return inserted

    async def run_once(
        self,
        *,
        now: datetime | None = None,
        force: bool = False,
        idle_seconds: float | None = None,
        power: PowerStatus | None = None,
        trace_bundles: Sequence[str | Path] = (),
    ) -> DreamRunResult:
        """Run one bounded reflection cycle."""

        started = now or datetime.now(UTC)
        if started.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        if power is None:
            power = await asyncio.to_thread(self.power_monitor.status)
        if self.config.require_plugged_in and (not power.known or not power.plugged_in):
            return DreamRunResult.skipped(
                "Dream Mode requires a known plugged-in power source", now=started
            )
        if not force:
            observed_idle = idle_seconds
            if observed_idle is None:
                observed_idle = await asyncio.to_thread(self.idle_monitor.idle_seconds)
            due, reason = self.scheduler.should_run(
                started,
                idle_seconds=observed_idle,
                power=power,
                require_plugged_in=self.config.require_plugged_in,
            )
            if not due:
                return DreamRunResult.skipped(reason, now=started)

        run_id = uuid.uuid4().hex
        tracer = AuditTracer(run_id=run_id, redactor=PIIRedactor())
        window = TraceWindow(
            start=started - timedelta(hours=self.config.lookback_hours),
            end=started,
        )
        result = DreamRunResult(
            run_id=run_id,
            status="completed",
            started_at=started.isoformat(),
            completed_at=started.isoformat(),
        )
        tracer.record(
            EventKind.SYSTEM_EVENT,
            {
                "stage": "dream_started",
                "window_start": window.start.isoformat(),
                "window_end": window.end.isoformat(),
                "tasks": list(self.config.tasks),
                "budget_tokens": self.config.budget_tokens,
            },
            actor="dream-mode",
        )
        try:
            records = await self._collect_records(window, trace_bundles)
            result.traces_replayed = len(records)
            result.estimated_tokens = estimate_tokens(
                trace_transcript(
                    records,
                    max_chars=self.config.max_trace_chars,
                )
            )
            tracer.record(
                EventKind.RETRIEVAL,
                {
                    "stage": "trace_replay",
                    "record_count": len(records),
                    "run_ids": sorted({record.run_id for record in records}),
                },
                actor="dream-mode",
            )
            if "distill_lessons" in self.config.tasks:
                result.lessons = await self._distill(records)
                tracer.record(
                    EventKind.DECISION,
                    {"stage": "lesson_distillation", "lesson_count": len(result.lessons)},
                    actor="dream-mode",
                )
            if "propose_memory_patches" in self.config.tasks:
                result.patch_ids = await self._propose_patches(result.lessons, records, run_id)
                tracer.record(
                    EventKind.MEMORY_WRITE,
                    {
                        "stage": "patch_proposals",
                        "patch_ids": result.patch_ids,
                        "approval_required": True,
                    },
                    actor="dream-mode",
                )
            if "consolidate_entities" in self.config.tasks and self.mesh is not None:
                result.mesh_reindexed, result.mesh_consolidations = await self._consolidate_mesh()
                tracer.record(
                    EventKind.STATE_CHANGE,
                    {
                        "stage": "mesh_consolidation",
                        "candidate_count": len(result.mesh_consolidations),
                    },
                    actor="dream-mode",
                )
            if "prune_stale" in self.config.tasks:
                result.stale_memories = await self._find_stale_memories(started)
                tracer.record(
                    EventKind.MEMORY_READ,
                    {"stage": "stale_memory_scan", "candidate_count": len(result.stale_memories)},
                    actor="dream-mode",
                )
        except Exception as exc:
            result.status = "failed"
            result.warnings.append(f"Dream Mode failed: {type(exc).__name__}: {exc}")
            tracer.record(
                EventKind.ERROR,
                {"stage": "dream_failed", "error": f"{type(exc).__name__}: {exc}"},
                actor="dream-mode",
            )

        result.completed_at = datetime.now(UTC).isoformat()
        tracer.record(
            EventKind.SYSTEM_EVENT,
            {
                "stage": "dream_completed",
                "status": result.status,
                "traces_replayed": result.traces_replayed,
            },
            actor="dream-mode",
        )
        result.audit_path = await self._export_audit(tracer, run_id)
        await asyncio.to_thread(self.state.save_run, result)
        return result

    async def run_forever(self) -> None:
        """Poll the schedule until :meth:`stop` is called or cancelled."""

        self._stop_event.clear()
        while not self._stop_event.is_set():
            now = datetime.now(UTC)
            power = await asyncio.to_thread(self.power_monitor.status)
            idle = await asyncio.to_thread(self.idle_monitor.idle_seconds)
            due, _ = self.scheduler.should_run(
                now,
                idle_seconds=idle,
                power=power,
                require_plugged_in=self.config.require_plugged_in,
            )
            if due:
                await self.run_once(now=now, force=True, idle_seconds=idle, power=power)
            await wait_for_stop(self._stop_event, self.config.poll_seconds)

    def morning_briefing(self, result: DreamRunResult | None = None) -> str:
        """Render a concise, actionable terminal briefing."""

        report = result or self.state.last_run()
        if report is None:
            return "Dream Mode: no run has completed."
        if report.status == "skipped":
            return f"Dream Mode skipped: {report.skipped_reason or 'not eligible'}."
        lines = [
            f"Dream Mode {report.status} ({report.completed_at})",
            f"  traces replayed: {report.traces_replayed}",
            f"  lessons distilled: {len(report.lessons)}",
            f"  memory patches proposed: {len(report.patch_ids)} (human review required)",
            f"  mesh reindexed: {'yes' if report.mesh_reindexed else 'no'}",
            f"  entity candidates: {len(report.mesh_consolidations)}",
            f"  stale memories flagged: {len(report.stale_memories)}",
        ]
        if report.audit_path:
            lines.append(f"  signed audit: {report.audit_path}")
        lines.extend(f"  warning: {warning}" for warning in report.warnings)
        return "\n".join(lines)

    async def _collect_records(
        self,
        window: TraceWindow,
        trace_bundles: Sequence[str | Path],
    ) -> list[AuditRecord]:
        records = await asyncio.to_thread(self.state.records, window)
        bundle_records: list[AuditRecord] = []
        for bundle_path in trace_bundles:
            loaded = await asyncio.to_thread(load_bundle, bundle_path)
            replay = await asyncio.to_thread(ReplayEngine().replay, bundle_path)
            if not replay.ok:
                raise ValueError(f"trace bundle failed replay validation: {bundle_path}")
            if any(
                window.start <= record.timestamp.astimezone(UTC) <= window.end
                for record in loaded.records
            ):
                bundle_records.extend(loaded.records)
        if bundle_records:
            await asyncio.to_thread(self.state.append_traces, bundle_records)
            await asyncio.to_thread(self.state.update_memory_reads, bundle_records)
            records.extend(bundle_records)
        unique = {record.event_id: record for record in records}
        return self._valid_trace_groups(list(unique.values()))

    @staticmethod
    def _order_chain(group: list[AuditRecord]) -> list[AuditRecord] | None:
        """Reconstruct a run's records in true hash-chain order.

        Records may arrive in any order (the store sorts by
        ``(timestamp, event_id)``, and two events sharing a timestamp are
        then ordered by a random ``event_id`` — which does *not* match the
        chain's ``prev_hash``/``hash`` linkage). ``verify_chain`` requires
        exact linkage order, so we relink here before verifying rather than
        trusting the incoming sort. Returns ``None`` if the group does not
        form a single, complete genesis-rooted chain (missing links, a
        fork, a cycle, or duplicate hashes).
        """

        by_prev: dict[str, AuditRecord] = {}
        for record in group:
            if record.prev_hash in by_prev:
                # Two records claim the same predecessor — a fork; not a
                # single verifiable chain.
                return None
            by_prev[record.prev_hash] = record

        ordered: list[AuditRecord] = []
        seen: set[str] = set()
        cursor = by_prev.get(GENESIS_HASH)
        while cursor is not None:
            if cursor.hash in seen:  # cycle guard
                return None
            ordered.append(cursor)
            seen.add(cursor.hash)
            cursor = by_prev.get(cursor.hash)

        if len(ordered) != len(group):
            return None
        return ordered

    @staticmethod
    def _valid_trace_groups(records: list[AuditRecord]) -> list[AuditRecord]:
        grouped: dict[str, list[AuditRecord]] = defaultdict(list)
        for record in records:
            grouped[record.run_id].append(record)
        valid: list[AuditRecord] = []
        for run_id, group in grouped.items():
            ordered = DreamMode._order_chain(group)
            if ordered is None:
                logger.warning(
                    "dream: dropping run %s — records do not form a single "
                    "verifiable hash chain (%d records)",
                    run_id,
                    len(group),
                )
                continue
            try:
                AuditTracer.verify_chain(ordered)
            except Exception as exc:
                logger.warning(
                    "dream: dropping run %s — chain verification failed: %s",
                    run_id,
                    exc,
                )
                continue
            valid.extend(ordered)
        return sorted(valid, key=lambda record: (record.timestamp, record.event_id))

    async def _distill(self, records: list[AuditRecord]) -> list[Lesson]:
        bounded = trace_transcript(
            records,
            max_chars=self.config.max_trace_chars,
        )
        if estimate_tokens(bounded) > self.config.budget_tokens:
            return await asyncio.to_thread(DeterministicLessonDistiller().distill, records)
        if self.backend is not None:
            distiller = ModelLessonDistiller(self.backend)
            lessons, consumed = await distiller.distill(
                records,
                max_chars=self.config.max_trace_chars,
            )
            if consumed > self.config.budget_tokens:
                return await asyncio.to_thread(DeterministicLessonDistiller().distill, records)
            return lessons
        return await asyncio.to_thread(DeterministicLessonDistiller().distill, records)

    async def _propose_patches(
        self,
        lessons: list[Lesson],
        records: list[AuditRecord],
        run_id: str,
    ) -> list[str]:
        if self.memory is None or not lessons:
            return []
        evidence = trace_transcript(
            records, max_chars=min(self.config.max_trace_chars // 2, self.config.budget_tokens * 2)
        )
        lesson_text = "\n".join(
            f"- {lesson.text} (theme={lesson.theme}, confidence={lesson.confidence:.2f}, "
            f"evidence={','.join(lesson.evidence_event_ids)})"
            for lesson in lessons
        )
        transcript = f"Dream Mode lessons:\n{lesson_text}\n\nTrace evidence:\n{evidence}"
        patches = await self.memory.propose_from_session(run_id, transcript=transcript)
        return [patch.patch_id for patch in patches]

    async def _consolidate_mesh(self) -> tuple[bool, list[MeshConsolidation]]:
        if self.mesh is None:
            return False, []
        await self.mesh.reindex()
        matches = await asyncio.to_thread(self.mesh.duplicates, limit=20)
        candidates: list[MeshConsolidation] = []
        for match in matches:
            left = str(getattr(match, "left_path", getattr(match, "path_a", "")))
            right = str(getattr(match, "right_path", getattr(match, "path_b", "")))
            score = float(getattr(match, "score", 0.0))
            if left and right:
                candidates.append(
                    MeshConsolidation(
                        left_path=left,
                        right_path=right,
                        score=score,
                        reason="KnowledgeMesh duplicate candidate; no automatic merge performed",
                    )
                )
        return True, candidates

    async def _find_stale_memories(self, now: datetime) -> list[StaleMemory]:
        paths = self.memory_paths
        if self.memory is not None and not paths:
            paths = tuple(self.memory.managed_paths)
        return await asyncio.to_thread(
            self.state.stale_memories,
            paths,
            now=now,
            stale_after_days=self.config.stale_after_days,
        )

    async def _export_audit(self, tracer: AuditTracer, run_id: str) -> str | None:
        output = Path(self.config.audit_dir).expanduser() / f"{run_id}.audit.zip"
        await asyncio.to_thread(
            export_audit_bundle,
            list(tracer.records),
            self.signing_policy,
            output,
        )
        return str(output)


def load_dream_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON report written by a caller from a local path."""

    return cast(dict[str, Any], json.loads(Path(path).expanduser().read_text(encoding="utf-8")))
