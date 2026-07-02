"""Self-improving agent wrapper with eval-gated config patches."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from .._compat import run_sync
from ..evaluation import EvalSuite, EvalSuiteResult, PromptOptimizer
from ..training import ABTestRouter, AutoRolloutManager, FeedbackCollector, RolloutPolicy
from ..training.types import FeedbackSample
from .executor import AgentExecutor
from .facade import SimpleAgent

PatchType = Literal[
    "prompt_rewrite", "tool_addition", "tool_removal", "routing_rule", "few_shot_examples"
]
PatchStatus = Literal[
    "proposed",
    "blocked",
    "approved",
    "canary",
    "promoted",
    "rolled_back",
    "rejected",
]


@dataclass
class AgentConfigSnapshot:
    """Serializable snapshot of the mutable agent config surface."""

    system_prompt: str
    tool_names: list[str] = field(default_factory=list)
    agent_type: str = "react"
    max_iterations: int = 10
    routing_rules: dict[str, Any] = field(default_factory=dict)
    few_shot_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentConfigPatch:
    """Signed, reversible patch proposed by an agent evolution cycle."""

    patch_type: PatchType
    description: str
    changes: dict[str, Any]
    patch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    author: str = "self-improving-agent"
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    eval_score: float | None = None
    baseline_score: float | None = None
    eval_cost_usd: float = 0.0
    eval_latency_ms: float = 0.0
    status: PatchStatus = "proposed"
    requires_human_approval: bool = False
    approved_by: str | None = None
    rollout_pct: float = 0.0
    rollback_of: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    signature: str = ""

    def sign(self, secret: str = "") -> str:
        """Update and return a deterministic SHA-256 patch signature."""
        material = json.dumps(
            self._signature_payload(),
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        self.signature = hashlib.sha256(f"{secret}:{material}".encode()).hexdigest()
        return self.signature

    def verify(self, secret: str = "") -> bool:
        """Verify the patch signature against current patch content."""
        material = json.dumps(
            self._signature_payload(),
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        expected = hashlib.sha256(f"{secret}:{material}".encode()).hexdigest()
        return self.signature == expected

    def to_dict(self) -> dict[str, Any]:
        if not self.signature:
            self.sign()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfigPatch:
        return cls(**data)

    def _signature_payload(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("signature", None)
        return data


@dataclass
class EvolutionCycleResult:
    """Outcome of one self-improvement cycle."""

    patch: AgentConfigPatch | None
    status: PatchStatus
    reason: str | None = None
    eval_result: EvalSuiteResult | None = None


@runtime_checkable
class MetaAnalyzerProtocol(Protocol):
    async def propose(
        self,
        *,
        samples: list[FeedbackSample],
        snapshot: AgentConfigSnapshot,
        improvement_targets: list[str],
    ) -> list[AgentConfigPatch]: ...


class MetaAnalyzer:
    """Analyze production feedback and propose deterministic config patches.

    If an LLM is supplied, it is asked for JSON patch suggestions. If parsing
    fails or no LLM is provided, a conservative heuristic proposes prompt
    rewrites from negative feedback and corrected responses.
    """

    def __init__(
        self,
        llm: Any | None = None,
        *,
        max_samples: int = 50,
        max_candidates: int = 3,
    ) -> None:
        self.llm = llm
        self.max_samples = max_samples
        self.max_candidates = max_candidates

    async def propose(
        self,
        *,
        samples: list[FeedbackSample],
        snapshot: AgentConfigSnapshot,
        improvement_targets: list[str],
    ) -> list[AgentConfigPatch]:
        recent = samples[-self.max_samples :]
        patches = await self._llm_propose(recent, snapshot, improvement_targets)
        if patches:
            return patches[: self.max_candidates]
        return self._heuristic_propose(recent, snapshot, improvement_targets)

    async def _llm_propose(
        self,
        samples: list[FeedbackSample],
        snapshot: AgentConfigSnapshot,
        improvement_targets: list[str],
    ) -> list[AgentConfigPatch]:
        if self.llm is None:
            return []

        prompt = self._build_analysis_prompt(samples, snapshot, improvement_targets)
        try:
            raw = await self.llm.generate(prompt)
        except Exception:
            return []

        proposals = _parse_json_objects(raw)
        patches: list[AgentConfigPatch] = []
        for item in proposals:
            patch_type = str(item.get("patch_type") or "prompt_rewrite")
            if patch_type not in _PATCH_TYPES:
                continue
            changes = item.get("changes")
            if not isinstance(changes, dict):
                continue
            patches.append(
                AgentConfigPatch(
                    patch_type=patch_type,  # type: ignore[arg-type]
                    description=str(item.get("description") or "LLM-proposed improvement"),
                    changes=changes,
                    metadata={"source": "meta_llm"},
                )
            )
        return patches

    def _heuristic_propose(
        self,
        samples: list[FeedbackSample],
        snapshot: AgentConfigSnapshot,
        improvement_targets: list[str],
    ) -> list[AgentConfigPatch]:
        if "prompt" not in improvement_targets and "prompt_rewrite" not in improvement_targets:
            return []

        negative = [s for s in samples if s.feedback == "negative"]
        if not negative:
            return []

        corrected = [s.corrected_response for s in negative if s.corrected_response]
        failure_notes = _summarize_failures(negative)
        guidance = [
            snapshot.system_prompt.strip(),
            "",
            "Improvement guidance from production feedback:",
        ]
        if failure_notes:
            guidance.extend(f"- {note}" for note in failure_notes)
        if corrected:
            guidance.append("- Prefer patterns from corrected responses when they are available.")
        guidance.append("- Ask for clarification when the request is ambiguous.")

        return [
            AgentConfigPatch(
                patch_type="prompt_rewrite",
                description="Incorporate recurring negative feedback into the system prompt.",
                changes={"system_prompt": "\n".join(part for part in guidance if part is not None)},
                metadata={
                    "source": "heuristic",
                    "negative_samples": len(negative),
                    "corrected_samples": len(corrected),
                },
            )
        ]

    def _build_analysis_prompt(
        self,
        samples: list[FeedbackSample],
        snapshot: AgentConfigSnapshot,
        improvement_targets: list[str],
    ) -> str:
        payload = {
            "current_config": snapshot.to_dict(),
            "improvement_targets": improvement_targets,
            "feedback": [
                {
                    "query": sample.query,
                    "response": sample.response,
                    "feedback": sample.feedback,
                    "corrected_response": sample.corrected_response,
                    "metadata": sample.metadata or {},
                }
                for sample in samples
            ],
        }
        return (
            "Analyze this agent feedback and return a JSON list of candidate patches. "
            "Each item must contain patch_type, description, and changes. "
            "Allowed patch_type values are prompt_rewrite, routing_rule, "
            "few_shot_examples, tool_addition, tool_removal. "
            "Prefer small reversible patches.\n\n"
            f"{json.dumps(payload, sort_keys=True, default=str)}"
        )


class AgentEvolutionAuditLog:
    """Append-only audit log for agent configuration patches."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._patches: list[AgentConfigPatch] = []
        if self.path and self.path.exists():
            self._load()

    def append(self, patch: AgentConfigPatch) -> AgentConfigPatch:
        if not patch.signature:
            patch.sign()
        self._patches.append(patch)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(patch.to_dict(), sort_keys=True, default=str) + "\n")
        return patch

    def list(
        self,
        *,
        status: PatchStatus | None = None,
        limit: int | None = None,
    ) -> list[AgentConfigPatch]:
        records = [p for p in self._patches if status is None or p.status == status]
        records = list(reversed(records))
        return records[:limit] if limit is not None else records

    def get(self, patch_id: str) -> AgentConfigPatch | None:
        for patch in self._patches:
            if patch.patch_id == patch_id:
                return patch
        return None

    def _load(self) -> None:
        if self.path is None:
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._patches.append(AgentConfigPatch.from_dict(json.loads(line)))


class SelfImprovingAgent:
    """Wrap an agent with an eval-blocked self-improvement loop."""

    def __init__(
        self,
        base_agent: AgentExecutor | SimpleAgent | Any,
        eval_suite: EvalSuite | str | None = None,
        *,
        rollout: RolloutPolicy | AutoRolloutManager | None = None,
        feedback_collector: FeedbackCollector | None = None,
        optimizer: PromptOptimizer | None = None,
        meta_analyzer: MetaAnalyzerProtocol | None = None,
        improvement_targets: list[str] | None = None,
        cadence: str | int = "manual",
        agent_id: str = "default",
        audit_log: AgentEvolutionAuditLog | None = None,
        audit_path: str | Path | None = None,
        metrics: Any | None = None,
        signature_secret: str = "",
    ) -> None:
        self.base_agent = base_agent
        self.eval_suite = _coerce_eval_suite(eval_suite, rollout)
        self.feedback_collector = feedback_collector or FeedbackCollector()
        self.optimizer = optimizer
        self.meta_analyzer = meta_analyzer or MetaAnalyzer()
        self.improvement_targets = improvement_targets or [
            "prompt",
            "tool_selection",
            "routing",
            "few_shot_examples",
        ]
        self.cadence = cadence
        self.agent_id = agent_id
        self.audit_log = audit_log or AgentEvolutionAuditLog(audit_path)
        self.metrics = metrics
        self.signature_secret = signature_secret
        self.routing_rules: dict[str, Any] = {}
        self.few_shot_examples: list[str] = []
        self._run_count = 0
        self._active_patch_ids: list[str] = []
        self._owned_rollout_router: ABTestRouter | None = None

        if isinstance(rollout, AutoRolloutManager):
            self.rollout_manager = rollout
            self.rollout_policy = rollout.policy
        else:
            self.rollout_policy = rollout or RolloutPolicy()
            self._owned_rollout_router = ABTestRouter(
                base_model=f"{agent_id}:base",
                finetuned_model=f"{agent_id}:candidate",
                rollout_pct=0.0,
                experiment_id=f"{agent_id}:evolution",
            )
            self.rollout_manager = AutoRolloutManager(
                router=self._owned_rollout_router,
                policy=self.rollout_policy,
            )

    async def arun(self, prompt: str, **kwargs: Any) -> str:
        """Run the wrapped agent asynchronously and trigger cadence checks."""
        started = time.perf_counter()
        answer = await _run_agent_async(self.base_agent, prompt, kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        feedback = kwargs.get("feedback")
        if feedback in {"positive", "negative"}:
            self.feedback_collector.record(
                prompt,
                answer,
                feedback,
                corrected_response=kwargs.get("corrected_response"),
                metadata={"agent_id": self.agent_id, "source": "self_improving_agent"},
                latency_ms=latency_ms,
                cost_usd=kwargs.get("cost_usd"),
            )
        self._run_count += 1
        if self._cadence_due():
            await self.evolve()
        return answer

    def run(self, prompt: str, **kwargs: Any) -> str:
        """Run the wrapped agent synchronously."""
        return str(run_sync(self.arun(prompt, **kwargs)))

    async def evolve(self, *, require_approval: bool | None = None) -> EvolutionCycleResult:
        """Execute one observe-diagnose-propose-validate-canary cycle."""
        samples = await self.feedback_collector.get_samples()
        snapshot = self._snapshot()
        proposals = await self.meta_analyzer.propose(
            samples=samples,
            snapshot=snapshot,
            improvement_targets=self.improvement_targets,
        )
        self._record_metric(
            "record_agent_evolution_cycle", agent_id=self.agent_id, proposals=len(proposals)
        )

        if not proposals:
            return EvolutionCycleResult(
                patch=None, status="rejected", reason="no candidate patches"
            )

        baseline = await self.eval_suite.score_prompt(snapshot.system_prompt)
        for patch in proposals:
            patch.metadata.setdefault("agent_id", self.agent_id)
            patch.before = snapshot.to_dict()
            patch.after = self._preview_patch(snapshot, patch).to_dict()
            patch.baseline_score = baseline.score
            patch.requires_human_approval = self.rollout_policy.requires_human_approval(
                patch.patch_type
            )
            patch.sign(self.signature_secret)

            candidate_result = await self._validate_patch(patch)
            patch.eval_score = candidate_result.score
            patch.eval_cost_usd = candidate_result.cost_usd
            patch.eval_latency_ms = candidate_result.latency_ms
            passed, reason = self.rollout_policy.check_eval_gate(
                candidate_result.score,
                baseline_score=baseline.score,
            )
            if not candidate_result.passed:
                passed = False
                reason = "; ".join(candidate_result.failures) or reason

            if not passed:
                patch.status = "blocked"
                patch.metadata["block_reason"] = reason
                patch.sign(self.signature_secret)
                self.audit_log.append(patch)
                self._record_patch_metric(patch)
                continue

            approval_required = patch.requires_human_approval
            if require_approval is not None:
                approval_required = require_approval
                patch.requires_human_approval = require_approval

            if approval_required and patch.approved_by is None:
                patch.status = "approved"
                patch.metadata["approval_required"] = True
                patch.sign(self.signature_secret)
                self.audit_log.append(patch)
                self._record_patch_metric(patch)
                return EvolutionCycleResult(
                    patch=patch,
                    status=patch.status,
                    reason="human approval required",
                    eval_result=candidate_result,
                )

            self._apply_patch(patch)
            patch.status = "canary"
            patch.rollout_pct = self.rollout_policy.initial_rollout_pct()
            patch.sign(self.signature_secret)
            self.audit_log.append(patch)
            self._active_patch_ids.append(patch.patch_id)
            self.rollout_manager.activate()
            self._record_patch_metric(patch)
            return EvolutionCycleResult(patch=patch, status="canary", eval_result=candidate_result)

        return EvolutionCycleResult(
            patch=None,
            status="blocked",
            reason="all candidate patches failed eval gates",
        )

    def approve(self, patch_id: str, *, approved_by: str = "human") -> AgentConfigPatch:
        """Approve and apply a previously eval-approved patch."""
        patch = self.audit_log.get(patch_id)
        if patch is None:
            raise KeyError(f"patch {patch_id!r} not found")
        if patch.status != "approved":
            raise ValueError(f"patch {patch_id!r} is not waiting for approval")
        patch.approved_by = approved_by
        patch.requires_human_approval = False
        self._apply_patch(patch)
        patch.status = "canary"
        patch.rollout_pct = self.rollout_policy.initial_rollout_pct()
        patch.sign(self.signature_secret)
        self._active_patch_ids.append(patch.patch_id)
        self.rollout_manager.activate()
        self.audit_log.append(patch)
        self._record_patch_metric(patch)
        return patch

    def rollback(self, patch_id: str, *, reason: str = "manual") -> AgentConfigPatch:
        """Revert an applied patch by restoring its recorded ``before`` snapshot."""
        patch = self.audit_log.get(patch_id)
        if patch is None:
            raise KeyError(f"patch {patch_id!r} not found")
        if not patch.before:
            raise ValueError(f"patch {patch_id!r} has no reversible snapshot")

        before = AgentConfigSnapshot(**patch.before)
        self._restore_snapshot(before)
        self.rollout_manager.rollback(reason)

        rollback_patch = AgentConfigPatch(
            patch_type=patch.patch_type,
            description=f"Rollback of {patch.patch_id}: {reason}",
            changes={"restore_snapshot": patch.before},
            before=patch.after or self._snapshot().to_dict(),
            after=patch.before,
            baseline_score=patch.baseline_score,
            eval_score=patch.eval_score,
            status="rolled_back",
            rollback_of=patch.patch_id,
            metadata={"reason": reason, "agent_id": self.agent_id},
        )
        rollback_patch.sign(self.signature_secret)
        self.audit_log.append(rollback_patch)

        patch.status = "rolled_back"
        patch.sign(self.signature_secret)
        if patch.patch_id in self._active_patch_ids:
            self._active_patch_ids.remove(patch.patch_id)
        return rollback_patch

    def evolution_history(
        self,
        *,
        status: PatchStatus | None = None,
        limit: int | None = None,
    ) -> list[AgentConfigPatch]:
        """Return patch audit history, newest first."""
        return self.audit_log.list(status=status, limit=limit)

    def inspect_evolution(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return compact history rows for CLI and dashboards."""
        rows: list[dict[str, Any]] = []
        for patch in self.evolution_history(limit=limit):
            rows.append(
                {
                    "patch_id": patch.patch_id,
                    "created_at": patch.created_at,
                    "type": patch.patch_type,
                    "status": patch.status,
                    "eval_score": patch.eval_score,
                    "rollout_pct": patch.rollout_pct,
                    "description": patch.description,
                    "signature": patch.signature,
                }
            )
        return rows

    async def _validate_patch(self, patch: AgentConfigPatch) -> EvalSuiteResult:
        candidate_prompt = str(
            patch.after.get("system_prompt") or patch.before.get("system_prompt") or ""
        )
        if self.optimizer is not None and patch.patch_type == "prompt_rewrite":
            self.optimizer.variants = [candidate_prompt]
            best = await self.optimizer.run(
                str(patch.before.get("system_prompt") or ""),
                patch.description,
            )
            passed = best.score >= self.rollout_policy.min_eval_score
            return EvalSuiteResult(
                score=best.score,
                cost_usd=best.cost_usd,
                n_cases=int((best.metadata or {}).get("n_cases") or 0),
                passed=passed,
                case_scores=[float(v) for v in (best.metadata or {}).get("scores", [])],
                failures=[]
                if passed
                else [
                    f"{self.optimizer.metric} {best.score:.3f} < min {self.rollout_policy.min_eval_score:.3f}"
                ],
            )
        return await self.eval_suite.score_prompt(candidate_prompt)

    def _snapshot(self) -> AgentConfigSnapshot:
        executor = _extract_executor(self.base_agent)
        if executor is None:
            prompt = getattr(self.base_agent, "system_prompt", "You are a helpful AI assistant.")
            tools = getattr(self.base_agent, "tools", [])
            return AgentConfigSnapshot(
                system_prompt=str(prompt),
                tool_names=[getattr(tool, "name", str(tool)) for tool in tools],
                routing_rules=dict(self.routing_rules),
                few_shot_examples=list(self.few_shot_examples),
            )

        return AgentConfigSnapshot(
            system_prompt=executor.config.system_prompt,
            tool_names=[tool.name for tool in executor.config.tools],
            agent_type=executor.config.agent_type,
            max_iterations=executor.config.max_iterations,
            routing_rules=dict(self.routing_rules),
            few_shot_examples=list(self.few_shot_examples),
        )

    def _preview_patch(
        self,
        snapshot: AgentConfigSnapshot,
        patch: AgentConfigPatch,
    ) -> AgentConfigSnapshot:
        next_snapshot = AgentConfigSnapshot(**snapshot.to_dict())
        changes = patch.changes

        if "system_prompt" in changes:
            next_snapshot.system_prompt = str(changes["system_prompt"])
        if "append_prompt" in changes:
            next_snapshot.system_prompt = (
                next_snapshot.system_prompt.rstrip() + "\n" + str(changes["append_prompt"]).strip()
            )
        if "agent_type" in changes:
            next_snapshot.agent_type = str(changes["agent_type"])
        if "max_iterations" in changes:
            next_snapshot.max_iterations = int(changes["max_iterations"])
        if "routing_rules" in changes and isinstance(changes["routing_rules"], dict):
            next_snapshot.routing_rules.update(changes["routing_rules"])
        if "few_shot_examples" in changes and isinstance(changes["few_shot_examples"], list):
            next_snapshot.few_shot_examples.extend(str(v) for v in changes["few_shot_examples"])
        if "tool_name" in changes and patch.patch_type == "tool_addition":
            tool_name = str(changes["tool_name"])
            if tool_name not in next_snapshot.tool_names:
                next_snapshot.tool_names.append(tool_name)
        if "tool_name" in changes and patch.patch_type == "tool_removal":
            tool_name = str(changes["tool_name"])
            next_snapshot.tool_names = [
                name for name in next_snapshot.tool_names if name != tool_name
            ]

        return next_snapshot

    def _apply_patch(self, patch: AgentConfigPatch) -> None:
        after = AgentConfigSnapshot(**patch.after)
        self._restore_snapshot(after)

    def _restore_snapshot(self, snapshot: AgentConfigSnapshot) -> None:
        executor = _extract_executor(self.base_agent)
        self.routing_rules = dict(snapshot.routing_rules)
        self.few_shot_examples = list(snapshot.few_shot_examples)

        if executor is None:
            if hasattr(self.base_agent, "system_prompt"):
                self.base_agent.system_prompt = snapshot.system_prompt
            return

        tools_by_name = {tool.name: tool for tool in executor.config.tools}
        executor.config.system_prompt = snapshot.system_prompt
        executor.config.agent_type = snapshot.agent_type  # type: ignore[assignment]
        executor.config.max_iterations = snapshot.max_iterations
        executor.config.tools = [
            tools_by_name[name] for name in snapshot.tool_names if name in tools_by_name
        ]
        executor.rebuild()

    def _cadence_due(self) -> bool:
        if self.cadence == "manual":
            return False
        if isinstance(self.cadence, int):
            return self.cadence > 0 and self._run_count % self.cadence == 0
        if isinstance(self.cadence, str) and self.cadence.startswith("every_"):
            parts = self.cadence.split("_")
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].startswith("run"):
                n = int(parts[1])
                return n > 0 and self._run_count % n == 0
        return False

    def _record_metric(self, method: str, **kwargs: Any) -> None:
        if self.metrics is None:
            return
        recorder = getattr(self.metrics, method, None)
        if recorder is None:
            return
        try:
            recorder(**kwargs)
        except Exception:
            return

    def _record_patch_metric(self, patch: AgentConfigPatch) -> None:
        self._record_metric(
            "record_agent_evolution_patch",
            agent_id=self.agent_id,
            patch_type=patch.patch_type,
            status=patch.status,
            eval_score=patch.eval_score,
            rollout_pct=patch.rollout_pct,
        )


_PATCH_TYPES = {
    "prompt_rewrite",
    "tool_addition",
    "tool_removal",
    "routing_rule",
    "few_shot_examples",
}


def _coerce_eval_suite(
    eval_suite: EvalSuite | str | None,
    rollout: RolloutPolicy | AutoRolloutManager | None,
) -> EvalSuite:
    threshold = None
    if isinstance(rollout, AutoRolloutManager):
        threshold = rollout.policy.min_eval_score
    elif isinstance(rollout, RolloutPolicy):
        threshold = rollout.min_eval_score
    if isinstance(eval_suite, EvalSuite):
        if eval_suite.threshold is None and threshold is not None:
            eval_suite.threshold = threshold
        return eval_suite
    if isinstance(eval_suite, str):
        return EvalSuite.from_decorators(eval_suite, threshold=threshold)
    return EvalSuite.from_cases([], threshold=threshold)


async def _run_agent_async(agent: Any, prompt: str, kwargs: dict[str, Any]) -> str:
    run_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"feedback", "corrected_response", "cost_usd"}
    }
    if hasattr(agent, "arun"):
        result = agent.arun(prompt, **run_kwargs)
    elif isinstance(agent, AgentExecutor):
        result = agent.run(prompt)
    elif hasattr(agent, "run") and inspect.iscoroutinefunction(agent.run):
        result = agent.run(prompt, **run_kwargs)
    elif hasattr(agent, "run"):
        return str(agent.run(prompt, **run_kwargs))
    else:
        raise TypeError("base_agent must expose run(), arun(), or AgentExecutor.run()")
    return str(await result)


def _extract_executor(agent: Any) -> AgentExecutor | None:
    if isinstance(agent, AgentExecutor):
        return agent
    executor = getattr(agent, "_executor", None)
    return executor if isinstance(executor, AgentExecutor) else None


def _parse_json_objects(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _summarize_failures(samples: list[FeedbackSample]) -> list[str]:
    notes: list[str] = []
    for sample in samples[:5]:
        query = sample.query.strip().replace("\n", " ")
        if len(query) > 120:
            query = query[:117] + "..."
        if query:
            notes.append(f"Handle requests like '{query}' more accurately.")
    return notes
