"""The :class:`GuardrailPolicy` -- one policy surface over input/output guards.

A policy runs its guards in order, rolls their findings into a
:class:`GuardrailReport`, records a signed, replayable audit entry for every
decision, and publishes a ``guardrail.*`` event to SynapseKit Live. Neither the
audit payload nor the Live event ever carries the raw text -- only guard names,
counts, the chosen action, and a provenance-tagged verdict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Guard, GuardContext
from .types import GuardFinding, GuardrailReport, Mode, Stage, strongest_mode

if TYPE_CHECKING:
    from ..audit import AuditTracer, SigningPolicy


class GuardrailPolicy:
    """Compose guards over the input and output stages.

    Guards may be passed pre-sorted (``input_guards`` / ``output_guards``) or as
    a single ``guards`` list that is routed by each guard's ``stage``. Every
    :meth:`check_input` / :meth:`check_output` call emits a signed audit record
    (unless ``audit=False``) and a Live event (unless ``live=False``).
    """

    def __init__(
        self,
        *,
        guards: list[Guard] | None = None,
        input_guards: list[Guard] | None = None,
        output_guards: list[Guard] | None = None,
        tracer: AuditTracer | None = None,
        audit: bool = True,
        live: bool = True,
        run_id: str | None = None,
    ) -> None:
        self.input_guards: list[Guard] = list(input_guards or [])
        self.output_guards: list[Guard] = list(output_guards or [])
        for guard in guards or []:
            (self.output_guards if guard.stage is Stage.OUTPUT else self.input_guards).append(guard)

        self._live = live
        if tracer is None and audit:
            from ..audit import AuditTracer

            tracer = AuditTracer(run_id=run_id)
        self.tracer = tracer

    # -- public API -------------------------------------------------------
    async def check_input(self, text: str, context: GuardContext | None = None) -> GuardrailReport:
        """Run the input guards over ``text`` and return the rolled-up report."""
        return await self._run(self.input_guards, text, Stage.INPUT, context)

    async def check_output(self, text: str, context: GuardContext | None = None) -> GuardrailReport:
        """Run the output guards over ``text`` and return the rolled-up report."""
        return await self._run(self.output_guards, text, Stage.OUTPUT, context)

    def export_audit_bundle(self, path: str, signing_policy: SigningPolicy | None = None) -> str:
        """Drain the audit trail and write a signed, verifiable bundle to ``path``.

        ``signing_policy`` is a :class:`~synapsekit.audit.SigningPolicy`; when
        omitted a fresh Ed25519 policy is generated. Returns ``path``."""
        if self.tracer is None:
            raise RuntimeError("policy has auditing disabled; no trail to export")
        from ..audit import SigningPolicy, export_audit_bundle

        signing = signing_policy if signing_policy is not None else SigningPolicy.ed25519()
        export_audit_bundle(self.tracer.drain(), signing, path)
        return path

    # -- internals --------------------------------------------------------
    async def _run(
        self,
        guards: list[Guard],
        text: str,
        stage: Stage,
        context: GuardContext | None,
    ) -> GuardrailReport:
        ctx = context or GuardContext(stage=stage)
        findings: list[GuardFinding] = []
        current = text
        triggered_modes: list[Mode] = []

        for guard in guards:
            finding = await guard.inspect(current, ctx)
            findings.append(finding)
            if not finding.triggered:
                continue
            triggered_modes.append(finding.mode)
            # A redact-mode hit rewrites the text seen by later guards.
            if finding.mode is Mode.REDACT and finding.redacted_text is not None:
                current = finding.redacted_text

        action = strongest_mode(triggered_modes)
        allowed = action not in (Mode.BLOCK, Mode.REQUIRE_HUMAN)
        report = GuardrailReport(
            stage=stage, text=current, allowed=allowed, action=action, findings=findings
        )
        self._emit(report)
        return report

    def _emit(self, report: GuardrailReport) -> None:
        triggered = report.triggered
        action_label = report.action.value if report.action else "allow"

        if self.tracer is not None:
            from ..provenance import GroundedSignal

            # Guardrail verdicts are heuristic and self-attested -- tag them as
            # self-reported so downstream consumers never mistake them for an
            # externally-grounded signal.
            signal = GroundedSignal.self_reported(
                0.0 if not report.allowed else 1.0,
                verdict=action_label,
                stage=report.stage.value,
            )
            payload = {
                "stage": report.stage.value,
                "action": action_label,
                "allowed": report.allowed,
                "guards": [f.guard for f in report.findings],
                "violations": [
                    {
                        "guard": f.guard,
                        "mode": f.mode.value,
                        "detail": f.detail,
                        "count": f.count,
                    }
                    for f in triggered
                ],
                "signal": signal.to_dict(),
            }
            kind = "GUARDRAIL_VIOLATION" if triggered else "GUARDRAIL_CHECK"
            record = self.tracer.record(kind, payload, actor="guardrails")
            report.audit_event_id = record.event_id

        if self._live:
            from ..live.bus import bus

            if bus.enabled:
                bus.publish(
                    {
                        "kind": "guardrail.violation" if triggered else "guardrail.check",
                        "name": f"guardrail.{report.stage.value}",
                        "status": "blocked" if not report.allowed else "ok",
                        "attributes": {
                            "stage": report.stage.value,
                            "action": action_label,
                            # rule names + counts only -- never the inspected text.
                            "rule": ", ".join(f.guard for f in triggered) or "none",
                            "violation_count": len(triggered),
                            "audit_event_id": report.audit_event_id,
                        },
                    }
                )
