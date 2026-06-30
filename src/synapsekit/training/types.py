"""Shared data types for the continuous fine-tuning pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uid() -> str:
    return str(uuid.uuid4())


@dataclass
class FeedbackSample:
    """Captured production feedback from a single user interaction."""

    query: str
    response: str
    feedback: Literal["positive", "negative"]
    id: str = field(default_factory=_uid)
    corrected_response: str | None = None
    context: list[str] | None = None
    metadata: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=_now)
    latency_ms: float | None = None
    cost_usd: float | None = None


@dataclass
class TrainingExample:
    """Single example in OpenAI / Anthropic chat JSONL format."""

    messages: list[dict[str, str]]
    source_feedback_id: str | None = None


@dataclass
class PreferencePair:
    """Chosen / rejected pair for DPO or RLHF training."""

    prompt: str
    chosen: str
    rejected: str
    source_ids: tuple[str, str] = field(default_factory=lambda: ("", ""))


@dataclass
class FineTuneJob:
    """Lifecycle descriptor for a provider fine-tuning job."""

    job_id: str
    provider: str
    base_model: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    fine_tuned_model: str | None = None
    error: str | None = None
    training_file_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ABTestResult:
    """Single observed result from an A/B test assignment."""

    user_id: str
    model_variant: Literal["base", "finetuned"]
    latency_ms: float
    cost_usd: float
    timestamp: datetime = field(default_factory=_now)
    feedback: Literal["positive", "negative"] | None = None
    eval_score: float | None = None


@dataclass
class ABTestMetrics:
    """Aggregate metrics for both A/B variants over a time window."""

    base_sample_count: int
    finetuned_sample_count: int
    base_latency_ms: float
    finetuned_latency_ms: float
    base_cost_usd: float
    finetuned_cost_usd: float
    base_positive_rate: float
    finetuned_positive_rate: float
    base_eval_score: float | None = None
    finetuned_eval_score: float | None = None

    @property
    def latency_delta_pct(self) -> float:
        """Positive = fine-tuned is slower than base."""
        if self.base_latency_ms == 0:
            return 0.0
        return (self.finetuned_latency_ms - self.base_latency_ms) / self.base_latency_ms * 100

    @property
    def quality_delta_pct(self) -> float:
        """Positive = fine-tuned has a higher user positive rate."""
        base = self.base_positive_rate
        if base == 0:
            return 100.0 if self.finetuned_positive_rate > 0 else 0.0
        return (self.finetuned_positive_rate - base) / base * 100

    @property
    def cost_delta_pct(self) -> float:
        """Positive = fine-tuned costs more per request."""
        if self.base_cost_usd == 0:
            return 0.0
        return (self.finetuned_cost_usd - self.base_cost_usd) / self.base_cost_usd * 100


@dataclass
class RolloutPolicy:
    """Configuration controlling how the rollout progresses and when it rolls back."""

    stages: list[float] = field(default_factory=lambda: [5.0, 25.0, 50.0, 100.0])
    min_samples_per_stage: int = 100
    improvement_threshold_pct: float = 2.0
    latency_regression_pct: float = 20.0
    cost_regression_pct: float = 15.0
    canary_pct: float | None = None
    min_eval_score: float = 0.85
    rollback_on_regression: bool = True
    require_human_approval_for: list[str] = field(default_factory=lambda: ["tool_addition"])

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("stages must contain at least one rollout percentage")
        for pct in self.stages:
            if not 0.0 <= pct <= 100.0:
                raise ValueError(f"rollout stages must be in [0, 100], got {pct}")
        if self.canary_pct is not None and not 0.0 <= self.canary_pct <= 100.0:
            raise ValueError(f"canary_pct must be in [0, 100], got {self.canary_pct}")
        if self.min_samples_per_stage < 0:
            raise ValueError("min_samples_per_stage must be >= 0")
        if not 0.0 <= self.min_eval_score <= 1.0:
            raise ValueError("min_eval_score must be in [0, 1]")

    def initial_rollout_pct(self) -> float:
        """Return the first rollout percentage, honoring explicit canary policy."""
        return self.rollout_stages()[0]

    def rollout_stages(self) -> list[float]:
        """Return the effective rollout stage sequence."""
        if self.canary_pct is None:
            return list(self.stages)
        return [self.canary_pct, *[pct for pct in self.stages if pct != self.canary_pct]]

    def requires_human_approval(self, change_type: str) -> bool:
        """Return whether a change type requires approval before it can ship."""
        return change_type in set(self.require_human_approval_for)

    def check_eval_gate(
        self,
        score: float,
        *,
        baseline_score: float | None = None,
    ) -> tuple[bool, str | None]:
        """Validate eval score and regression policy for an evolution patch."""
        if score < self.min_eval_score:
            return False, f"eval score {score:.3f} < required {self.min_eval_score:.3f}"
        if self.rollback_on_regression and baseline_score is not None and score < baseline_score:
            return False, f"eval score regressed from {baseline_score:.3f} to {score:.3f}"
        return True, None


@dataclass
class RolloutState:
    """Mutable state snapshot of the active rollout."""

    current_pct: float = 0.0
    stage_idx: int = 0
    stage_samples: int = 0
    stage_start_finetuned_count: int = 0
    is_active: bool = False
    rolled_back: bool = False
    rollback_reason: str | None = None


@dataclass
class CostBenefitReport:
    """Financial and quality summary of a fine-tuning investment."""

    training_cost_usd: float
    monthly_inference_savings_usd: float
    quality_improvement_pct: float
    estimated_payback_days: float
    metadata: dict[str, Any] | None = None
