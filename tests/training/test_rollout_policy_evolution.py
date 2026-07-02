from __future__ import annotations

import pytest

from synapsekit.training import ABTestRouter, AutoRolloutManager, RolloutPolicy


def test_rollout_policy_canary_pct_overrides_first_stage():
    policy = RolloutPolicy(stages=[10.0, 50.0], canary_pct=5.0)
    router = ABTestRouter("base", "candidate", rollout_pct=0.0)
    manager = AutoRolloutManager(router=router, policy=policy)

    manager.activate()

    assert router.rollout_pct == 5.0
    assert manager.state.current_pct == 5.0


def test_rollout_policy_canary_pct_preserves_following_stages():
    policy = RolloutPolicy(stages=[10.0, 50.0], canary_pct=5.0)

    assert policy.rollout_stages() == [5.0, 10.0, 50.0]


def test_rollout_policy_requires_human_approval_by_change_type():
    policy = RolloutPolicy(require_human_approval_for=["tool_addition", "prompt_rewrite"])

    assert policy.requires_human_approval("tool_addition") is True
    assert policy.requires_human_approval("routing_rule") is False


def test_rollout_policy_blocks_low_eval_score():
    policy = RolloutPolicy(min_eval_score=0.85)

    passed, reason = policy.check_eval_gate(0.7)

    assert passed is False
    assert reason == "eval score 0.700 < required 0.850"


def test_rollout_policy_blocks_regression_when_enabled():
    policy = RolloutPolicy(min_eval_score=0.5, rollback_on_regression=True)

    passed, reason = policy.check_eval_gate(0.8, baseline_score=0.9)

    assert passed is False
    assert reason == "eval score regressed from 0.900 to 0.800"


def test_rollout_policy_allows_regression_when_disabled():
    policy = RolloutPolicy(min_eval_score=0.5, rollback_on_regression=False)

    passed, reason = policy.check_eval_gate(0.8, baseline_score=0.9)

    assert passed is True
    assert reason is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stages": []}, "stages must contain at least one rollout percentage"),
        ({"stages": [101.0]}, "rollout stages must be in"),
        ({"canary_pct": 101.0}, "canary_pct must be in"),
        ({"min_samples_per_stage": -1}, "min_samples_per_stage must be >= 0"),
        ({"min_eval_score": 1.1}, "min_eval_score must be in"),
    ],
)
def test_rollout_policy_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RolloutPolicy(**kwargs)


def test_rollout_policy_canary_pct_zero_prepends_zero_stage():
    """canary_pct=0.0 must prepend a 0-pct canary stage before the regular stages."""
    policy = RolloutPolicy(stages=[10.0, 50.0], canary_pct=0.0)

    stages = policy.rollout_stages()

    assert stages == [0.0, 10.0, 50.0]
    assert stages[0] == 0.0


def test_rollout_policy_canary_pct_zero_activates_at_zero_pct():
    """Activating a rollout with canary_pct=0.0 sets the router to 0% traffic."""
    policy = RolloutPolicy(stages=[10.0, 50.0], canary_pct=0.0)
    router = ABTestRouter("base", "candidate", rollout_pct=0.0)
    manager = AutoRolloutManager(router=router, policy=policy)

    manager.activate()

    assert router.rollout_pct == 0.0
    assert manager.state.current_pct == 0.0
