"""Explicit, bounded bilateral price/SLA negotiation.

This module records agreement only; it does not execute a task or enforce it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from threading import Lock


class MessageKind(str, Enum):
    OFFER = "offer"
    COUNTER = "counter"
    ACCEPT = "accept"
    REJECT = "reject"


class NegotiationStatus(str, Enum):
    OPEN = "open"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, whitespace-trimmed string.")


def _decimal(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal, not a float or integer.")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite.")


@dataclass(frozen=True, slots=True)
class Terms:
    """Price per task, upper latency in milliseconds and minimum availability in percent."""

    price: Decimal
    currency: str
    max_latency_ms: Decimal
    min_availability_percent: Decimal
    availability_window_seconds: Decimal

    def __post_init__(self) -> None:
        for name in (
            "price",
            "max_latency_ms",
            "min_availability_percent",
            "availability_window_seconds",
        ):
            _decimal(getattr(self, name), name)
        if self.price < 0 or self.max_latency_ms <= 0 or self.availability_window_seconds <= 0:
            raise ValueError("Price must be nonnegative; latency and window must be positive.")
        if not Decimal(0) <= self.min_availability_percent <= Decimal(100):
            raise ValueError("Availability must be between 0 and 100 percent.")
        if not isinstance(self.currency, str) or re.fullmatch(r"[A-Z]{3}", self.currency) is None:
            raise ValueError("Currency must be a three-letter uppercase code.")


@dataclass(frozen=True, slots=True)
class Message:
    negotiation_id: str
    message_id: str
    sender: str
    recipient: str
    kind: MessageKind
    terms: Terms | None
    in_reply_to: str | None = None

    def __post_init__(self) -> None:
        for name in ("negotiation_id", "message_id", "sender", "recipient"):
            _identifier(getattr(self, name), name)
        if self.sender == self.recipient:
            raise ValueError("Sender and recipient must differ.")
        if not isinstance(self.kind, MessageKind):
            raise TypeError("kind must be a MessageKind.")
        if self.in_reply_to is not None:
            _identifier(self.in_reply_to, "in_reply_to")
        if self.kind == MessageKind.REJECT:
            if self.terms is not None:
                raise ValueError("A rejection has no terms.")
        elif not isinstance(self.terms, Terms):
            raise TypeError("Offer, counter and accept require Terms.")


class NegotiationRole(str, Enum):
    BUYER = "buyer"
    SELLER = "seller"


@dataclass(frozen=True, slots=True)
class BoundedPolicy:
    """Accept within reservation bounds or concede directly to those bounds.

    A buyer's reservation is maximum price/latency and minimum availability;
    a seller's is minimum price/latency and maximum availability. Currency and
    availability window must match exactly.
    """

    role: NegotiationRole
    reservation: Terms

    def __post_init__(self) -> None:
        if not isinstance(self.role, NegotiationRole):
            raise TypeError("role must be a NegotiationRole.")
        if not isinstance(self.reservation, Terms):
            raise TypeError("reservation must be Terms.")

    def accepts(self, terms: Terms) -> bool:
        if not isinstance(terms, Terms):
            raise TypeError("terms must be Terms.")
        bound = self.reservation
        if (
            terms.currency != bound.currency
            or terms.availability_window_seconds != bound.availability_window_seconds
        ):
            return False
        if self.role == NegotiationRole.BUYER:
            return (
                terms.price <= bound.price
                and terms.max_latency_ms <= bound.max_latency_ms
                and terms.min_availability_percent >= bound.min_availability_percent
            )
        return (
            terms.price >= bound.price
            and terms.max_latency_ms >= bound.max_latency_ms
            and terms.min_availability_percent <= bound.min_availability_percent
        )

    def respond(self, proposal: Message, *, message_id: str, can_counter: bool) -> Message:
        """Respond to a proposal; the caller tracks whether the round cap permits a counter."""
        if not isinstance(proposal, Message) or proposal.kind not in (
            MessageKind.OFFER,
            MessageKind.COUNTER,
        ):
            raise ValueError("A policy can only respond to an offer or counter.")
        if not isinstance(can_counter, bool):
            raise TypeError("can_counter must be a bool.")
        assert proposal.terms is not None
        if self.accepts(proposal.terms):
            kind, response_terms = MessageKind.ACCEPT, proposal.terms
        elif (
            can_counter
            and proposal.terms.currency == self.reservation.currency
            and proposal.terms.availability_window_seconds
            == self.reservation.availability_window_seconds
        ):
            kind, response_terms = MessageKind.COUNTER, self.reservation
        else:
            kind, response_terms = MessageKind.REJECT, None
        return Message(
            proposal.negotiation_id,
            message_id,
            proposal.recipient,
            proposal.sender,
            kind,
            response_terms,
            proposal.message_id,
        )


class Negotiation:
    """In-memory transcript; responses reference the latest proposal.

    ``max_rounds`` counts proposals, including the initial offer. Acceptance
    or rejection is still allowed after the final proposal.
    """

    def __init__(self, initial: Message, *, max_rounds: int = 10) -> None:
        if not isinstance(initial, Message) or initial.kind != MessageKind.OFFER:
            raise ValueError("Initial message must be an offer.")
        if initial.in_reply_to is not None:
            raise ValueError("Initial offer cannot be a reply.")
        if type(max_rounds) is not int or not 1 <= max_rounds <= 1000:
            raise ValueError("max_rounds must be an integer between 1 and 1000.")
        self._lock = Lock()
        self._messages = [initial]
        self._ids = {initial.message_id}
        self._max_rounds = max_rounds
        self._proposals = 1
        self._status = NegotiationStatus.OPEN
        self._agreement: Terms | None = None

    @property
    def status(self) -> NegotiationStatus:
        with self._lock:
            return self._status

    @property
    def agreement(self) -> Terms | None:
        with self._lock:
            return self._agreement

    @property
    def messages(self) -> tuple[Message, ...]:
        with self._lock:
            return tuple(self._messages)

    def apply(self, message: Message) -> None:
        if not isinstance(message, Message):
            raise TypeError("Response must be a Message.")
        with self._lock:
            self._apply_locked(message)

    def _apply_locked(self, message: Message) -> None:
        if self._status != NegotiationStatus.OPEN:
            raise ValueError("Negotiation is already closed.")
        previous = self._messages[-1]
        if (
            message.negotiation_id != previous.negotiation_id
            or message.message_id in self._ids
            or message.in_reply_to != previous.message_id
            or message.sender != previous.recipient
            or message.recipient != previous.sender
        ):
            raise ValueError("Response has invalid parties, id, or stale reply reference.")
        if message.kind == MessageKind.OFFER:
            raise ValueError("Only the initial message can be an offer.")
        if message.kind == MessageKind.COUNTER:
            if self._proposals >= self._max_rounds:
                raise ValueError("Negotiation proposal limit reached.")
            assert message.terms is not None and previous.terms is not None
            if message.terms.currency != previous.terms.currency:
                raise ValueError("Currency cannot change during a negotiation.")
        elif message.kind == MessageKind.ACCEPT and message.terms != previous.terms:
            raise ValueError("Acceptance must exactly match the latest terms.")

        self._messages.append(message)
        self._ids.add(message.message_id)
        if message.kind == MessageKind.COUNTER:
            self._proposals += 1
        elif message.kind == MessageKind.ACCEPT:
            self._status = NegotiationStatus.ACCEPTED
            self._agreement = message.terms
        else:
            self._status = NegotiationStatus.REJECTED
