"""High-level Agent OS Shell session orchestration."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from synapsekit.audit import AuditTracer, EventKind, PIIRedactor, export_audit_bundle

from .context import ContextCollector, NullAmbientContext
from .executor import DirectShellExecutor, git_diff
from .history import ShellHistory
from .lexer import lex_input
from .planner import CachedPlanner, PlanningError, RuleBasedPlanner, ShellPlanner, TranslationCache
from .safety import SafetyAnalyzer, SafetyError, SafetyPolicy
from .types import (
    CommandResult,
    PlannedStep,
    SegmentKind,
    ShellContext,
    ShellPlan,
    ShellRunResult,
)

ConfirmCallback = Callable[[PlannedStep, str], Awaitable[bool]]


class ShellSession:
    """Plan, preview, confirm, execute, and attest one shell interaction."""

    def __init__(
        self,
        *,
        planner: ShellPlanner | None = None,
        cwd: str | Path | None = None,
        shell: str | None = None,
        mesh: Any | None = None,
        ambient: Any | None = None,
        history: ShellHistory | None = None,
        safety_policy: SafetyPolicy | None = None,
        signing_policy: Any | None = None,
        audit_dir: str | Path | None = None,
        timeout: float = 30.0,
        max_output_bytes: int = 1_000_000,
    ) -> None:
        self.history = history or ShellHistory()
        self.context_collector = ContextCollector(
            cwd=cwd,
            shell=shell,
            mesh=mesh,
            ambient=ambient or NullAmbientContext(),
            history=self.history,
        )
        self.planner = planner or CachedPlanner(
            RuleBasedPlanner(), TranslationCache(), model="rules"
        )
        self.safety = SafetyAnalyzer(safety_policy)
        self.signing_policy = signing_policy
        self.audit_dir = Path(audit_dir).expanduser() if audit_dir else None
        self.executor = DirectShellExecutor(timeout=timeout, max_output_bytes=max_output_bytes)
        self.tracer = AuditTracer(run_id=uuid.uuid4().hex, redactor=PIIRedactor())
        self.last_context: ShellContext | None = None

    async def plan(self, input_text: str) -> ShellPlan:
        if not input_text.strip():
            raise PlanningError("input is empty")
        context = await self.context_collector.collect(input_text)
        self.last_context = context
        self.tracer.record(
            EventKind.USER_INPUT,
            {"input": input_text, "cwd": context.cwd, "shell": context.shell.value},
            actor="user",
        )
        if context.mesh_hits:
            self.tracer.record(
                EventKind.RETRIEVAL, {"query": input_text, "hits": list(context.mesh_hits)}
            )
        self.tracer.record(EventKind.STATE_CHANGE, {"context": context.to_dict()}, actor="shell")

        parsed = lex_input(input_text)
        steps: list[PlannedStep] = []
        warnings: list[str] = []
        for segment in parsed.segments:
            if segment.kind is SegmentKind.SHELL:
                if segment.text.strip(" &|;\t\r\n"):
                    steps.append(
                        PlannedStep(
                            segment.text.strip(), "Execute the supplied shell text.", "shell"
                        )
                    )
                continue
            try:
                planned = await self.planner.plan(segment.text, context)
            except PlanningError:
                raise
            steps.extend(planned)
        if not steps:
            raise PlanningError("no shell or natural-language steps were found")
        for step in steps:
            assessment = self.safety.assess(step.command)
            if assessment.destructive:
                warnings.append(assessment.preview)
        return ShellPlan(
            input_text=input_text,
            steps=tuple(steps),
            summary="; ".join(step.explanation or step.command for step in steps),
            warnings=tuple(warnings),
        )

    async def run(
        self,
        input_text: str,
        *,
        confirm: ConfirmCallback | None = None,
        assume_yes: bool = False,
        dry_run: bool = False,
    ) -> ShellRunResult:
        try:
            plan = await self.plan(input_text)
        except (PlanningError, SafetyError) as exc:
            return ShellRunResult(
                plan=ShellPlan(input_text, (), "", ()), commands=(), aborted=True, error=str(exc)
            )

        destructive_steps: list[tuple[PlannedStep, str]] = []
        for step in plan.steps:
            assessment = self.safety.assess(step.command)
            if assessment.destructive:
                destructive_steps.append((step, assessment.preview))
        if destructive_steps:
            if not self.safety.policy.allow_destructive:
                return await self._abort(plan, "destructive commands are disabled by policy")
            if self.safety.policy.require_signed_approval and self.signing_policy is None:
                return await self._abort(
                    plan,
                    "destructive commands require --signing-key; unsigned execution is disabled",
                )
            for step, preview in destructive_steps:
                approved = assume_yes
                if not approved and confirm is not None:
                    approved = await confirm(step, preview)
                if not approved:
                    self.tracer.record(
                        EventKind.DECISION,
                        {"command": step.command, "approved": False, "preview": preview},
                        actor="user",
                    )
                    audit_path = await self._export_audit("denied")
                    return ShellRunResult(
                        plan,
                        (),
                        aborted=True,
                        error="destructive command not approved",
                        audit_path=audit_path,
                    )
                self.tracer.record(
                    EventKind.DECISION,
                    {"command": step.command, "approved": True, "preview": preview},
                    actor="user",
                )
            # This export is the signed pre-execution receipt. The final
            # export below adds tool results and the post-execution diff.
            audit_path = await self._export_audit("preflight")
        else:
            audit_path = None

        if dry_run:
            return ShellRunResult(plan, (), audit_path=audit_path)

        results: list[CommandResult] = []
        for step in plan.steps:
            self.tracer.record(
                EventKind.TOOL_CALL, {"command": step.command, "source": step.source}, actor="shell"
            )
            step_results = await self.executor.run_text(
                step.command, self.last_context or await self.context_collector.collect()
            )
            results.extend(step_results)
            self.tracer.record(
                EventKind.TOOL_RESULT,
                {
                    "command": step.command,
                    "results": [
                        {
                            "exit_code": result.exit_code,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "timed_out": result.timed_out,
                            "skipped": result.skipped,
                        }
                        for result in step_results
                    ],
                },
                actor="shell",
            )
        if destructive_steps:
            self.tracer.record(
                EventKind.STATE_CHANGE,
                {"git_diff_stat": await git_diff(self.context_collector.cwd)},
                actor="shell",
            )
        ok = all(result.ok for result in results)
        await self.history.record(
            cwd=str(self.context_collector.cwd),
            input_text=input_text,
            commands=[step.command for step in plan.steps],
            ok=ok,
        )
        final_audit = (
            await self._export_audit("final") if self.signing_policy is not None else audit_path
        )
        return ShellRunResult(plan, tuple(results), audit_path=final_audit)

    async def _abort(self, plan: ShellPlan, message: str) -> ShellRunResult:
        self.tracer.record(EventKind.ERROR, {"error": message}, actor="shell")
        audit_path = await self._export_audit("abort")
        return ShellRunResult(plan, (), aborted=True, error=message, audit_path=audit_path)

    async def _export_audit(self, suffix: str) -> str | None:
        if self.signing_policy is None:
            return None
        directory = self.audit_dir or Path.home() / ".synapsekit" / "shell" / "audit"
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        path = directory / f"{self.tracer.run_id}-{suffix}.audit.zip"
        records = list(self.tracer.records)
        await asyncio.to_thread(export_audit_bundle, records, self.signing_policy, path)
        return str(path)


def result_to_dict(result: ShellRunResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "aborted": result.aborted,
        "error": result.error,
        "audit_path": result.audit_path,
        "plan": {
            "input": result.plan.input_text,
            "summary": result.plan.summary,
            "warnings": list(result.plan.warnings),
            "steps": [
                {
                    "command": step.command,
                    "explanation": step.explanation,
                    "source": step.source,
                    "confidence": step.confidence,
                }
                for step in result.plan.steps
            ],
        },
        "commands": [
            {
                "command": item.command,
                "stdout": item.stdout,
                "stderr": item.stderr,
                "exit_code": item.exit_code,
                "duration_seconds": item.duration_seconds,
                "timed_out": item.timed_out,
                "skipped": item.skipped,
            }
            for item in result.commands
        ],
    }
