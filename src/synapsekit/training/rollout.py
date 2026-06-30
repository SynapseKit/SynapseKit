"""Gradual rollout manager with automatic advancement and safety rollback."""

from __future__ import annotations

import logging
from typing import Literal

from .ab_testing import ABTestRouter
from .types import ABTestMetrics, RolloutPolicy, RolloutState

_log = logging.getLogger(__name__)


class AutoRolloutManager:
    """
    Manages gradual rollout of a fine-tuned model variant through configurable
    traffic percentage stages.

    Advancement rules
    -----------------
    * The fine-tuned model must show ``>= improvement_threshold_pct`` quality
      gain (positive user feedback rate delta) over the base model.
    * A minimum of ``min_samples_per_stage`` *new* fine-tuned observations must
      be collected *since the current stage started* before advancing or
      rolling back.  This prevents noise from a handful of observations
      triggering premature decisions.
    * Stages advance sequentially: 5 → 25 → 50 → 100 (configurable).

    Automatic rollback triggers (require min_samples_per_stage first)
    ---------------------------
    * Latency regression exceeds ``latency_regression_pct``.
    * Cost regression exceeds ``cost_regression_pct``.
    * Quality drops below ``-improvement_threshold_pct`` relative to base.

    Parameters
    ----------
    router:
        The ABTestRouter whose ``rollout_pct`` this manager controls.
    policy:
        Rollout policy.  Defaults to conservative 4-stage policy.
    """

    def __init__(
        self,
        router: ABTestRouter,
        policy: RolloutPolicy | None = None,
    ) -> None:
        self._router = router
        self._policy = policy or RolloutPolicy()
        self._state = RolloutState()

    @property
    def state(self) -> RolloutState:
        return self._state

    @property
    def policy(self) -> RolloutPolicy:
        return self._policy

    # ── Public API ────────────────────────────────────────────────────────────

    def activate(self, initial_finetuned_count: int = 0) -> None:
        """
        Begin rollout at the first stage percentage.

        Parameters
        ----------
        initial_finetuned_count:
            The number of fine-tuned results already recorded in the router
            at activation time.  Used as the per-stage baseline so that only
            *new* samples collected after activation count toward the
            ``min_samples_per_stage`` gate.
        """
        self._state.is_active = True
        self._state.rolled_back = False
        self._state.rollback_reason = None
        self._state.stage_idx = 0
        self._state.stage_samples = 0
        self._state.stage_start_finetuned_count = initial_finetuned_count
        self._state.current_pct = self._policy.initial_rollout_pct()
        self._router.rollout_pct = self._state.current_pct
        _log.info("Rollout activated at %.1f%%", self._state.current_pct)

    def evaluate_and_advance(
        self, metrics: ABTestMetrics
    ) -> Literal["advanced", "held", "rolled_back", "completed"]:
        """
        Evaluate current A/B metrics and decide whether to advance, hold,
        or roll back.

        Parameters
        ----------
        metrics:
            Aggregate A/B metrics for the current evaluation window.

        Returns
        -------
        "advanced"     — moved to the next rollout stage.
        "held"         — insufficient data or improvement below threshold.
        "rolled_back"  — automatic rollback triggered; router reset to 0%.
        "completed"    — reached 100% rollout.
        """
        if not self._state.is_active or self._state.rolled_back:
            return "held"

        if self._check_and_rollback(metrics):
            return "rolled_back"

        stage_new_samples = metrics.finetuned_sample_count - self._state.stage_start_finetuned_count
        if stage_new_samples < self._policy.min_samples_per_stage:
            return "held"

        if metrics.quality_delta_pct < self._policy.improvement_threshold_pct:
            return "held"

        return self._advance(metrics.finetuned_sample_count)

    def rollback(self, reason: str = "manual") -> None:
        """Immediately reset fine-tuned traffic to 0%."""
        self._state.is_active = False
        self._state.rolled_back = True
        self._state.rollback_reason = reason
        self._state.current_pct = 0.0
        self._router.rollout_pct = 0.0
        _log.warning("ROLLBACK — %s", reason)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_and_rollback(self, m: ABTestMetrics) -> bool:
        # Require minimum stage samples before any regression check so that
        # noise from a handful of observations cannot trigger irreversible
        # rollback.
        stage_new_samples = m.finetuned_sample_count - self._state.stage_start_finetuned_count
        if stage_new_samples < self._policy.min_samples_per_stage:
            return False

        if m.latency_delta_pct > self._policy.latency_regression_pct:
            self.rollback(
                f"latency regression {m.latency_delta_pct:.1f}% "
                f"> threshold {self._policy.latency_regression_pct:.1f}%"
            )
            return True
        if m.cost_delta_pct > self._policy.cost_regression_pct:
            self.rollback(
                f"cost regression {m.cost_delta_pct:.1f}% "
                f"> threshold {self._policy.cost_regression_pct:.1f}%"
            )
            return True
        if m.quality_delta_pct < -self._policy.improvement_threshold_pct:
            self.rollback(f"quality regression {m.quality_delta_pct:.1f}%")
            return True
        return False

    def _advance(self, current_finetuned_count: int) -> Literal["advanced", "completed"]:
        stages = self._policy.rollout_stages()
        next_idx = self._state.stage_idx + 1
        if next_idx >= len(stages):
            self._state.current_pct = 100.0
            self._router.rollout_pct = 100.0
            self._state.is_active = False
            _log.info("Rollout COMPLETED at 100%%")
            return "completed"

        self._state.stage_idx = next_idx
        self._state.stage_samples = 0
        self._state.stage_start_finetuned_count = current_finetuned_count
        self._state.current_pct = stages[next_idx]
        self._router.rollout_pct = self._state.current_pct
        _log.info(
            "Advanced to stage %d (%.1f%%)",
            next_idx,
            self._state.current_pct,
        )
        return "advanced"
