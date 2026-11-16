"""Seeded, offline-friendly multi-turn user simulation and a booking domain."""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

Actor = Literal["user", "agent"]
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
Observation_co = TypeVar("Observation_co", covariant=True)
Action_contra = TypeVar("Action_contra", contravariant=True)


class InvalidActionError(ValueError):
    """An actor attempted an unsupported or impermissible action."""


@dataclass(frozen=True)
class Reward:
    """A policy-derived score and recorded violations."""

    value: float
    violations: tuple[str, ...] = ()


class SimulationEnvironment(Protocol[Observation_co, Action_contra]):
    """Episode-owned shared state; observations should be independent snapshots."""

    async def reset(self, seed: int) -> None: ...

    async def observe(self, actor: Actor) -> Observation_co: ...

    async def apply(self, actor: Actor, action: Action_contra) -> bool:
        """Apply one action and return whether the episode is complete."""
        ...

    async def reward(self) -> Reward: ...


@dataclass(frozen=True)
class Turn(Generic[ObservationT, ActionT]):
    actor: Actor
    observation: ObservationT
    action: ActionT
    error: str | None = None


@dataclass(frozen=True)
class SimulationResult(Generic[ObservationT, ActionT]):
    seed: int
    reason: Literal["completed", "max_turns", "invalid_action"]
    turns: tuple[Turn[ObservationT, ActionT], ...]
    reward: Reward
    error: str | None = None


class SimulationHarness(Generic[ObservationT, ActionT]):
    """Alternate user and agent against one environment, resetting each run.

    The turn limit counts individual actions. InvalidActionError from apply is
    propagated by default or recorded and terminates the run. Observations and
    actions are deep-copied for the transcript before actor and apply callbacks,
    respectively. Copy failures and other exceptions propagate. Do not run a
    harness concurrently.
    """

    def __init__(
        self,
        environment: SimulationEnvironment[ObservationT, ActionT],
        user: Callable[[ObservationT, random.Random], Awaitable[ActionT]],
        agent: Callable[[ObservationT, random.Random], Awaitable[ActionT]],
        *,
        max_turns: int = 20,
        error_policy: Literal["raise", "terminate"] = "raise",
    ) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if error_policy not in ("raise", "terminate"):
            raise ValueError("error_policy must be 'raise' or 'terminate'")
        self.environment = environment
        self.user = user
        self.agent = agent
        self.max_turns = max_turns
        self.error_policy = error_policy

    async def run(self, *, seed: int) -> SimulationResult[ObservationT, ActionT]:
        await self.environment.reset(seed)
        # Independent streams prevent one actor's draws perturbing the other.
        user_rng = random.Random(f"user:{seed}")
        agent_rng = random.Random(f"agent:{seed}")
        turns: list[Turn[ObservationT, ActionT]] = []
        reason: Literal["completed", "max_turns", "invalid_action"] = "max_turns"
        error = None
        for index in range(self.max_turns):
            actor: Actor = "user" if index % 2 == 0 else "agent"
            observation = await self.environment.observe(actor)
            recorded_observation = deepcopy(observation)
            action = await (
                self.user(observation, user_rng)
                if actor == "user"
                else self.agent(observation, agent_rng)
            )
            recorded_action = deepcopy(action)
            try:
                done = await self.environment.apply(actor, action)
            except InvalidActionError as exc:
                if self.error_policy == "raise":
                    raise
                error = str(exc)
                turns.append(Turn(actor, recorded_observation, recorded_action, error))
                reason = "invalid_action"
                break
            turns.append(Turn(actor, recorded_observation, recorded_action))
            if done:
                reason = "completed"
                break
        return SimulationResult(seed, reason, tuple(turns), await self.environment.reward(), error)


@dataclass(frozen=True)
class BookingAction:
    """Action in the support-booking domain; kind determines relevant fields."""

    kind: str
    slot: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class BookingObservation:
    actor: Actor
    available_slots: tuple[str, ...]
    requested_slot: str | None
    consented: bool
    booked_slot: str | None
    last_message: str | None
    done: bool
    violations: tuple[str, ...]


class BookingEnvironment:
    """Shared support calendar with explicit consent before booking.

    Users request, consent, cancel, or wait; agents message, book, finish, or
    wait. A user can cancel a booking or change an unbooked request.
    """

    def __init__(self, slots: tuple[str, ...] = ("09:00", "10:00", "11:00")) -> None:
        if (
            not slots
            or any(not isinstance(slot, str) or not slot for slot in slots)
            or len(set(slots)) != len(slots)
        ):
            raise ValueError("slots must contain unique nonempty strings")
        self._slots = tuple(slots)
        self._clear()

    def _clear(self) -> None:
        self._requested: str | None = None
        self._consented = False
        self._booked: str | None = None
        self._message: str | None = None
        self._done = False
        self._violations: list[str] = []

    async def reset(self, seed: int) -> None:
        self._clear()

    async def observe(self, actor: Actor) -> BookingObservation:
        if actor not in ("user", "agent"):
            raise InvalidActionError(f"unknown actor: {actor!r}")
        return BookingObservation(
            actor,
            tuple(slot for slot in self._slots if slot != self._booked),
            self._requested,
            self._consented,
            self._booked,
            self._message,
            self._done,
            tuple(self._violations),
        )

    async def apply(self, actor: Actor, action: BookingAction) -> bool:
        if actor not in ("user", "agent") or not isinstance(action, BookingAction):
            raise InvalidActionError("invalid actor or action type")
        if self._done:
            raise InvalidActionError("episode is already complete")
        if action.kind == "wait":
            return False
        if actor == "user" and action.kind == "request":
            if self._booked is not None or action.slot not in self._slots:
                raise InvalidActionError("requested slot is unavailable")
            self._requested = action.slot
            self._consented = False
        elif actor == "user" and action.kind == "consent":
            if self._requested is None or self._booked is not None:
                raise InvalidActionError("consent requires an unbooked request")
            self._consented = True
        elif actor == "user" and action.kind == "cancel":
            if self._requested is None:
                raise InvalidActionError("nothing to cancel")
            self._requested = self._booked = None
            self._consented = False
        elif actor == "agent" and action.kind == "message":
            if not isinstance(action.message, str) or not action.message.strip():
                raise InvalidActionError("message must be nonempty")
            self._message = action.message
        elif actor == "agent" and action.kind == "book":
            if not self._consented:
                self._violations.append("booking_without_consent")
                raise InvalidActionError("booking requires user consent")
            if (
                self._booked is not None
                or action.slot != self._requested
                or action.slot not in self._slots
            ):
                self._violations.append("booking_wrong_slot")
                raise InvalidActionError("booking must match the available requested slot")
            self._booked = action.slot
        elif actor == "agent" and action.kind == "finish":
            if self._booked is None:
                self._violations.append("finish_without_booking")
                raise InvalidActionError("cannot finish without a booking")
            self._done = True
        else:
            raise InvalidActionError(f"unsupported {actor} action: {action.kind!r}")
        return self._done

    async def reward(self) -> Reward:
        success = (
            self._done
            and self._consented
            and self._booked is not None
            and self._booked == self._requested
            and not self._violations
        )
        return Reward(1.0 if success else 0.0, tuple(self._violations))
