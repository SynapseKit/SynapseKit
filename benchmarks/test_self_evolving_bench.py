"""CI-enforced gate for the #732 self-evolving acceptance criterion.

Runs one `SelfImprovingAgent` through 5 real evolution cycles and hard-enforces
that it improves itself, and that the eval gate — not the agent — decides:

* final held-out accuracy is near-perfect (>= 90%);
* it beats its own starting accuracy by >= 35 percentage points;
* held-out accuracy never regresses across the 5 cycles;
* the baseline is genuinely fallible (<= 50%), so the uplift is real, not trivial;
* the deliberately bad decoy candidates are *blocked* with a recorded reason;
* every accepted patch is reversible — rolling the last one back restores the
  previous held-out accuracy exactly — and every patch signature survives a
  JSONL round-trip through the audit log.
"""

from __future__ import annotations

import asyncio
from itertools import pairwise

import pytest
from self_evolving_bench import N_CYCLES, run_evolution

from synapsekit.agents import AgentEvolutionAuditLog

_MIN_FINAL_ACCURACY = 0.90
_MIN_UPLIFT = 0.35
_MAX_BASELINE_ACCURACY = 0.50
_ACCEPTED_STATUSES = {"canary", "promoted"}


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    audit_path = tmp_path_factory.mktemp("self_evolving_bench") / "evolution.jsonl"
    return asyncio.run(run_evolution(audit_path))


def test_final_accuracy_is_near_perfect(result):
    acc = result.final_score
    assert acc >= _MIN_FINAL_ACCURACY, (
        f"final held-out accuracy {acc:.0%} < {_MIN_FINAL_ACCURACY:.0%}"
    )


def test_uplift_over_baseline(result):
    assert result.uplift >= _MIN_UPLIFT, (
        f"baseline={result.baseline_score:.0%} vs final={result.final_score:.0%} → "
        f"uplift {result.uplift:.0%} < {_MIN_UPLIFT:.0%}"
    )


def test_accuracy_never_regresses(result):
    scores = [result.baseline_score] + [c.eval_score for c in result.cycles]
    for prev, curr in pairwise(scores):
        assert curr >= prev, (
            f"held-out accuracy regressed {prev:.0%} → {curr:.0%} across "
            f"{['baseline', *[f'cycle {c.cycle}' for c in result.cycles]]}"
        )


def test_baseline_is_actually_fallible(result):
    # Guard against a trivially-easy suite: the un-evolved prompt must miss some.
    acc = result.baseline_score
    assert acc <= _MAX_BASELINE_ACCURACY, (
        f"baseline too accurate ({acc:.0%} > {_MAX_BASELINE_ACCURACY:.0%}); "
        "the suite isn't testing anything"
    )


def test_decoy_candidates_are_blocked_by_the_eval_gate(result):
    blocked = [p for p in result.history if p.status == "blocked"]
    assert blocked, (
        f"0 blocked patches out of {len(result.history)}; the eval gate never "
        "rejected anything, so it is rubber-stamping"
    )
    for patch in blocked:
        reason = patch.metadata.get("block_reason")
        assert reason, (
            f"blocked patch {patch.patch_id} has block_reason={reason!r}, required a "
            "non-empty reason"
        )


def test_accepted_patches_are_eval_approved_and_reversible(result):
    accepted = result.accepted
    assert len(accepted) == N_CYCLES, (
        f"{len(accepted)} accepted patches, required {N_CYCLES} (one per cycle)"
    )
    for record in accepted:
        assert record.status in _ACCEPTED_STATUSES, (
            f"cycle {record.cycle} accepted with status {record.status!r}, "
            f"required one of {sorted(_ACCEPTED_STATUSES)}"
        )
        patch = result.agent.audit_log.get(record.patch_id)
        assert patch is not None, (
            f"cycle {record.cycle} patch {record.patch_id} missing from audit log"
        )
        assert patch.before, (
            f"cycle {record.cycle} patch {record.patch_id} has an empty `before` snapshot, "
            "so it is not reversible"
        )


def test_history_covers_every_cycle_and_survives_jsonl_round_trip(result):
    assert len(result.cycles) == N_CYCLES, (
        f"{len(result.cycles)} cycles recorded, required {N_CYCLES}"
    )
    known = {p.patch_id for p in result.history}
    for record in result.cycles:
        expected = [pid for pid in [record.patch_id, *record.blocked_patch_ids] if pid]
        assert expected, f"cycle {record.cycle} produced no audit entries at all"
        missing = [pid for pid in expected if pid not in known]
        assert not missing, (
            f"cycle {record.cycle} patches missing from evolution_history(): {missing}"
        )

    reloaded = AgentEvolutionAuditLog(result.audit_path).list()
    assert len(reloaded) == len(result.history), (
        f"round-tripped {len(reloaded)} patches, required {len(result.history)}"
    )
    unverified = [p.patch_id for p in reloaded if not p.verify()]
    assert not unverified, (
        f"{len(unverified)} of {len(reloaded)} patch signatures failed to verify after a "
        f"JSONL round-trip: {unverified}"
    )


def test_rollback_restores_the_previous_held_out_accuracy(result):
    # Ordered last: rollback mutates the agent's live config on purpose.
    last = result.accepted[-1]
    previous = result.cycles[-2].eval_score

    rollback = result.agent.rollback(last.patch_id, reason="bench rollback probe")
    restored = result.score_current_prompt()

    assert restored == pytest.approx(previous), (
        f"rollback of {last.patch_id} left held-out accuracy at {restored:.3f}, "
        f"required exactly the pre-patch {previous:.3f}"
    )
    assert rollback.status == "rolled_back", (
        f"rollback entry status {rollback.status!r}, required 'rolled_back'"
    )
    assert rollback.rollback_of == last.patch_id, (
        f"rollback entry points at {rollback.rollback_of!r}, required {last.patch_id!r}"
    )
    rolled_back = [p for p in result.agent.evolution_history() if p.status == "rolled_back"]
    assert rolled_back, "no rolled_back entry was appended to the audit log"
