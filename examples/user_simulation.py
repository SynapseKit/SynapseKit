"""Run an offline, seeded support-booking episode."""

from __future__ import annotations

import asyncio
import random

from synapsekit.evaluation.simulation import (
    BookingAction,
    BookingEnvironment,
    BookingObservation,
    SimulationHarness,
)


async def simulated_user(obs: BookingObservation, rng: random.Random) -> BookingAction:
    if obs.requested_slot is None:
        return BookingAction("request", slot=rng.choice(obs.available_slots))
    if not obs.consented:
        return BookingAction("consent")
    return BookingAction("wait")


async def support_agent(obs: BookingObservation, rng: random.Random) -> BookingAction:
    if not obs.consented:
        return BookingAction("message", message="May I book this support appointment?")
    if obs.booked_slot is None:
        return BookingAction("book", slot=obs.requested_slot)
    return BookingAction("finish", message="Your appointment is booked.")


async def main() -> None:
    harness = SimulationHarness(BookingEnvironment(), simulated_user, support_agent)
    result = await harness.run(seed=19)
    for turn in result.turns:
        print(f"{turn.actor}: {turn.action}")
    print(
        f"reason={result.reason} reward={result.reward.value} violations={result.reward.violations}"
    )


if __name__ == "__main__":
    asyncio.run(main())
