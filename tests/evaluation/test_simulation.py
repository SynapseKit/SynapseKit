from __future__ import annotations

import asyncio
import socket

import pytest


def run(coro):
    return asyncio.run(coro)


def test_shared_state_booking_succeeds_and_records_both_sides():
    from synapsekit.evaluation.simulation import (
        BookingAction,
        BookingEnvironment,
        SimulationHarness,
    )

    async def user(obs, rng):
        if obs.requested_slot is None:
            return BookingAction("request", slot=obs.available_slots[0])
        if not obs.consented:
            return BookingAction("consent")
        return BookingAction("wait")

    async def agent(obs, rng):
        if not obs.consented:
            return BookingAction("message", message="Please confirm your booking.")
        if obs.booked_slot is None:
            return BookingAction("book", slot=obs.requested_slot)
        return BookingAction("finish")

    result = run(SimulationHarness(BookingEnvironment(), user, agent).run(seed=7))
    assert result.reason == "completed"
    assert result.reward.value == 1.0
    assert result.reward.violations == ()
    assert [(t.actor, t.action.kind) for t in result.turns] == [
        ("user", "request"),
        ("agent", "message"),
        ("user", "consent"),
        ("agent", "book"),
        ("user", "wait"),
        ("agent", "finish"),
    ]
    assert result.turns[1].observation.requested_slot is not None
    assert result.turns[3].observation.consented is True
    assert result.turns[4].observation.booked_slot == result.turns[0].action.slot


def test_booking_without_consent_is_rejected_and_penalized():
    from synapsekit.evaluation.simulation import (
        BookingAction,
        BookingEnvironment,
        SimulationHarness,
    )

    async def user(obs, rng):
        return BookingAction("request", slot=obs.available_slots[0])

    async def agent(obs, rng):
        return BookingAction("book", slot=obs.requested_slot, message="I succeeded")

    result = run(
        SimulationHarness(BookingEnvironment(), user, agent, error_policy="terminate").run(seed=1)
    )
    assert result.reason == "invalid_action"
    assert result.reward.value == 0.0
    assert "booking_without_consent" in result.reward.violations
    assert result.error is not None
    assert result.turns[-1].action.kind == "book"


def test_requesting_unavailable_slot_is_rejected():
    from synapsekit.evaluation.simulation import (
        BookingAction,
        BookingEnvironment,
        InvalidActionError,
    )

    async def exercise():
        env = BookingEnvironment()
        await env.reset(0)
        await env.apply("user", BookingAction("request", slot="not-a-slot"))

    with pytest.raises(InvalidActionError):
        run(exercise())


def test_invalid_action_default_raises_and_limit_terminates():
    from synapsekit.evaluation.simulation import (
        BookingAction,
        BookingEnvironment,
        InvalidActionError,
        SimulationHarness,
    )

    async def invalid(obs, rng):
        return BookingAction("book", slot="nope")

    async def wait(obs, rng):
        return BookingAction("wait")

    with pytest.raises(InvalidActionError):
        run(SimulationHarness(BookingEnvironment(), invalid, wait).run(seed=2))
    result = run(SimulationHarness(BookingEnvironment(), wait, wait, max_turns=2).run(seed=2))
    assert result.reason == "max_turns"
    assert len(result.turns) == 2
    assert result.reward.value == 0.0
    with pytest.raises(ValueError):
        SimulationHarness(BookingEnvironment(), wait, wait, max_turns=0)
    with pytest.raises(ValueError):
        SimulationHarness(BookingEnvironment(), wait, wait, error_policy="ignore")


def test_seed_reproducibility_reset_and_offline(monkeypatch):
    from synapsekit.evaluation.simulation import (
        BookingAction,
        BookingEnvironment,
        SimulationHarness,
    )

    def no_network(*args, **kwargs):
        raise AssertionError("network access is not allowed")

    async def user(obs, rng):
        if obs.requested_slot is None:
            return BookingAction("request", slot=rng.choice(obs.available_slots))
        if not obs.consented:
            return BookingAction("consent")
        return BookingAction("wait")

    async def agent(obs, rng):
        if not obs.consented:
            return BookingAction("wait")
        if obs.booked_slot is None:
            return BookingAction("book", slot=obs.requested_slot)
        return BookingAction("finish")

    async def exercise():
        # Windows asyncio creates a loopback socketpair when the loop starts.
        with monkeypatch.context() as guard:
            guard.setattr(socket.socket, "connect", no_network)
            guard.setattr(socket, "create_connection", no_network)
            harness = SimulationHarness(BookingEnvironment(), user, agent)
            first = await harness.run(seed=19)
            second = await harness.run(seed=19)
            assert first == second
            assert first.reward.value == 1.0
            assert first.turns[0].observation.requested_slot is None
            assert first.turns[0].observation.booked_slot is None
            slots = {(await harness.run(seed=i)).turns[0].action.slot for i in range(10)}
            assert len(slots) > 1

    run(exercise())


def test_reward_ignores_claims_and_remembers_prior_policy_violation():
    from synapsekit.evaluation.simulation import (
        BookingAction,
        BookingEnvironment,
        InvalidActionError,
    )

    async def exercise():
        env = BookingEnvironment()
        await env.reset(3)
        obs = await env.observe("user")
        slot = obs.available_slots[0]
        await env.apply("user", BookingAction("request", slot=slot))
        with pytest.raises(InvalidActionError):
            await env.apply("agent", BookingAction("book", slot=slot))
        await env.apply("user", BookingAction("consent"))
        await env.apply("agent", BookingAction("book", slot=slot))
        await env.apply("agent", BookingAction("finish", message="Perfect!"))
        return await env.reward()

    reward = run(exercise())
    assert reward.value == 0.0
    assert "booking_without_consent" in reward.violations


def test_user_cancel_changes_agent_booking_and_resets_consent():
    from synapsekit.evaluation.simulation import BookingAction, BookingEnvironment

    async def exercise():
        env = BookingEnvironment()
        await env.reset(0)
        initial = await env.observe("user")
        slot = initial.available_slots[0]
        await env.apply("user", BookingAction("request", slot=slot))
        await env.apply("user", BookingAction("consent"))
        await env.apply("agent", BookingAction("book", slot=slot))
        booked = await env.observe("user")
        assert booked.booked_slot == slot and slot not in booked.available_slots
        await env.apply("user", BookingAction("cancel"))
        cancelled = await env.observe("agent")
        assert cancelled.booked_slot is None and cancelled.requested_slot is None
        assert not cancelled.consented and slot in cancelled.available_slots
        assert booked.booked_slot == slot  # old observation remains unchanged
        assert (await env.reward()).value == 0.0

    run(exercise())


def test_wrong_slot_after_consent_records_violation():
    from synapsekit.evaluation.simulation import (
        BookingAction,
        BookingEnvironment,
        InvalidActionError,
    )

    async def exercise():
        env = BookingEnvironment()
        await env.reset(0)
        slots = (await env.observe("user")).available_slots
        await env.apply("user", BookingAction("request", slot=slots[0]))
        await env.apply("user", BookingAction("consent"))
        with pytest.raises(InvalidActionError):
            await env.apply("agent", BookingAction("book", slot=slots[1]))
        assert (await env.observe("user")).booked_slot is None
        assert (await env.reward()).violations == ("booking_wrong_slot",)

    run(exercise())


def test_actor_exception_is_not_swallowed_by_terminate_policy():
    from synapsekit.evaluation.simulation import (
        BookingAction,
        BookingEnvironment,
        SimulationHarness,
    )

    async def broken(obs, rng):
        raise RuntimeError("actor error")

    async def wait(obs, rng):
        return BookingAction("wait")

    with pytest.raises(RuntimeError, match="actor error"):
        run(
            SimulationHarness(BookingEnvironment(), broken, wait, error_policy="terminate").run(
                seed=0
            )
        )


def test_custom_environment_contract():
    from synapsekit.evaluation.simulation import Reward, SimulationHarness

    class Counter:
        async def reset(self, seed):
            self.value = seed

        async def observe(self, actor):
            return (actor, self.value)

        async def apply(self, actor, action):
            self.value += action
            return self.value >= 5

        async def reward(self):
            return Reward(float(self.value))

    async def actor(obs, rng):
        return 1

    result = run(SimulationHarness(Counter(), actor, actor).run(seed=3))
    assert result.reason == "completed"
    assert [turn.observation for turn in result.turns] == [("user", 3), ("agent", 4)]
    assert result.reward.value == 5.0


def test_mutable_observations_are_snapshotted_before_actor_changes_them():
    from synapsekit.evaluation.simulation import Reward, SimulationHarness

    class Environment:
        async def reset(self, seed):
            self.observation = {"nested": {"n": 0}}

        async def observe(self, actor):
            return self.observation

        async def apply(self, actor, action):
            assert self.observation["nested"]["n"] == 99
            self.observation["nested"]["n"] = 100
            return True

        async def reward(self):
            return Reward(0.0)

    async def actor(observation, rng):
        observation["nested"]["n"] = 99
        return "done"

    result = run(SimulationHarness(Environment(), actor, actor).run(seed=0))
    assert result.turns[0].observation == {"nested": {"n": 0}}


@pytest.mark.parametrize("invalid", [False, True])
def test_mutable_actions_are_snapshotted_before_apply_and_reuse(invalid):
    from synapsekit.evaluation.simulation import InvalidActionError, Reward, SimulationHarness

    shared = {"nested": {"n": 0}}

    class Environment:
        async def reset(self, seed):
            self.count = 0

        async def observe(self, actor):
            return self.count

        async def apply(self, actor, action):
            self.count += 1
            action["nested"]["n"] = 99
            if invalid and self.count == 2:
                raise InvalidActionError("rejected")
            return self.count == 2

        async def reward(self):
            return Reward(0.0)

    async def actor(observation, rng):
        shared["nested"]["n"] = observation
        return shared

    result = run(
        SimulationHarness(Environment(), actor, actor, error_policy="terminate").run(seed=0)
    )
    shared["nested"]["n"] = 500
    assert [turn.action for turn in result.turns] == [{"nested": {"n": 0}}, {"nested": {"n": 1}}]
    assert result.reason == ("invalid_action" if invalid else "completed")
    assert result.turns[-1].error == ("rejected" if invalid else None)


@pytest.mark.parametrize("uncopyable", ["observation", "action"])
def test_snapshot_copy_failure_propagates_before_next_callback(uncopyable):
    from synapsekit.evaluation.simulation import Reward, SimulationHarness

    events = []

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot snapshot")

    class Environment:
        async def reset(self, seed):
            pass

        async def observe(self, actor):
            return Uncopyable() if uncopyable == "observation" else 0

        async def apply(self, actor, action):
            events.append("apply")
            return True

        async def reward(self):
            return Reward(0.0)

    async def actor(observation, rng):
        events.append("actor")
        return Uncopyable()

    with pytest.raises(RuntimeError, match="cannot snapshot"):
        run(SimulationHarness(Environment(), actor, actor, error_policy="terminate").run(seed=0))
    assert events == ([] if uncopyable == "observation" else ["actor"])
