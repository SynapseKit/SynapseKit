import asyncio
import threading
from dataclasses import replace
from decimal import Decimal

import pytest

from synapsekit.agents.agent_registry import AgentMetadata
from synapsekit.agents.federation import AgentFederation
from synapsekit.agents.negotiation import (
    Message,
    MessageKind,
    Negotiation,
    NegotiationStatus,
    Terms,
)


def terms(price="10.00", latency="200", availability="99.9", currency="USD"):
    return Terms(Decimal(price), currency, Decimal(latency), Decimal(availability), Decimal("3600"))


def offer():
    return Message("neg-1", "msg-1", "buyer", "seller", MessageKind.OFFER, terms())


def test_offer_counter_accept_converges_with_explicit_agreement():
    session = Negotiation(offer(), max_rounds=3)
    counter = Message(
        "neg-1",
        "msg-2",
        "seller",
        "buyer",
        MessageKind.COUNTER,
        terms("12.50", "150", "99.95"),
        "msg-1",
    )
    session.apply(counter)
    accepted = Message(
        "neg-1", "msg-3", "buyer", "seller", MessageKind.ACCEPT, counter.terms, "msg-2"
    )
    session.apply(accepted)
    assert session.status == NegotiationStatus.ACCEPTED
    assert session.agreement == counter.terms
    assert session.messages == (offer(), counter, accepted)


def test_reject_stale_replayed_and_wrong_party_messages():
    session = Negotiation(offer(), max_rounds=3)
    counter = Message(
        "neg-1", "msg-2", "seller", "buyer", MessageKind.COUNTER, terms("12"), "msg-1"
    )
    session.apply(counter)
    for bad in [
        counter,
        replace(counter, message_id="msg-3", sender="seller"),
        replace(
            counter, message_id="msg-3", sender="buyer", recipient="seller", in_reply_to="msg-1"
        ),
        replace(
            counter,
            message_id="msg-3",
            sender="buyer",
            recipient="seller",
            negotiation_id="other",
            in_reply_to="msg-2",
        ),
    ]:
        with pytest.raises(ValueError):
            session.apply(bad)
    assert session.messages == (offer(), counter)


def test_accept_must_match_latest_offer_and_terminal_state():
    session = Negotiation(offer())
    bad = Message("neg-1", "msg-2", "seller", "buyer", MessageKind.ACCEPT, terms("9"), "msg-1")
    with pytest.raises(ValueError):
        session.apply(bad)
    accepted = replace(bad, terms=offer().terms)
    session.apply(accepted)
    with pytest.raises(ValueError):
        session.apply(accepted)


def test_concurrent_terminal_responses_are_atomic():
    session = Negotiation(offer())
    start = threading.Barrier(3)
    first_at_previous = threading.Event()
    release_first = threading.Event()
    competitor_at_boundary = threading.Event()
    marker_lock = threading.Lock()
    outcomes = []

    class PausingMessages(list):
        def __getitem__(self, index):
            value = super().__getitem__(index)
            with marker_lock:
                first = not first_at_previous.is_set()
                if first:
                    first_at_previous.set()
            if first and not release_first.wait(10):
                raise RuntimeError("first response was not released")
            return value

    # Observe contention without relying on scheduling delays. On an unlocked
    # implementation, the competing worker signals when it finishes instead.
    class ObservedLock:
        def __init__(self, lock):
            self.lock = lock

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                competitor_at_boundary.set()
                self.lock.acquire()
            return self

        def __exit__(self, *args):
            self.lock.release()

    session._messages = PausingMessages(session._messages)
    if hasattr(session, "_lock"):
        session._lock = ObservedLock(session._lock)

    def respond(message):
        start.wait(timeout=10)
        try:
            session.apply(message)
            outcomes.append((message, None))
        except Exception as exc:
            outcomes.append((message, exc))
        finally:
            competitor_at_boundary.set()

    accepted = Message(
        "neg-1", "accept", "seller", "buyer", MessageKind.ACCEPT, offer().terms, "msg-1"
    )
    rejected = Message("neg-1", "reject", "seller", "buyer", MessageKind.REJECT, None, "msg-1")
    threads = [
        threading.Thread(target=respond, args=(message,)) for message in (accepted, rejected)
    ]
    for thread in threads:
        thread.start()
    try:
        start.wait(timeout=10)
        assert first_at_previous.wait(10), "neither response reached the previous proposal"
        assert competitor_at_boundary.wait(10), "competing response did not reach the race boundary"
    finally:
        release_first.set()
        for thread in threads:
            thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert len(outcomes) == 2
    successes = [message for message, error in outcomes if error is None]
    errors = [error for _, error in outcomes if error is not None]
    assert len(successes) == 1
    assert len(errors) == 1 and isinstance(errors[0], ValueError)
    winner = successes[0]
    assert session.messages == (offer(), winner)
    assert session.status == (
        NegotiationStatus.ACCEPTED
        if winner.kind == MessageKind.ACCEPT
        else NegotiationStatus.REJECTED
    )
    assert session.agreement == (winner.terms if winner.kind == MessageKind.ACCEPT else None)


def test_currency_cannot_change_and_bad_initial_offers_are_rejected():
    session = Negotiation(offer())
    counter = Message(
        "neg-1", "msg-2", "seller", "buyer", MessageKind.COUNTER, terms(currency="EUR"), "msg-1"
    )
    with pytest.raises(ValueError, match="Currency"):
        session.apply(counter)
    assert session.messages == (offer(),)
    with pytest.raises(ValueError):
        Negotiation(replace(offer(), in_reply_to="earlier"))
    for invalid in (0, -1, True, 1001):
        with pytest.raises(ValueError):
            Negotiation(offer(), max_rounds=invalid)


def test_round_limit_and_explicit_no_overlap():
    session = Negotiation(offer(), max_rounds=1)
    counter = Message(
        "neg-1", "msg-2", "seller", "buyer", MessageKind.COUNTER, terms("20"), "msg-1"
    )
    with pytest.raises(ValueError):
        session.apply(counter)
    reject = replace(counter, kind=MessageKind.REJECT, terms=None)
    session.apply(reject)
    assert session.status == NegotiationStatus.REJECTED
    assert session.agreement is None


@pytest.mark.parametrize(
    "price,currency,latency,availability",
    [
        (Decimal("NaN"), "USD", Decimal("1"), Decimal("99")),
        (Decimal("-1"), "USD", Decimal("1"), Decimal("99")),
        (1.1, "USD", Decimal("1"), Decimal("99")),
        (Decimal("1"), "usd", Decimal("1"), Decimal("99")),
        (Decimal("1"), "USD", Decimal("0"), Decimal("99")),
        (Decimal("1"), "USD", Decimal("1"), Decimal("101")),
    ],
)
def test_invalid_terms(price, currency, latency, availability):
    with pytest.raises((TypeError, ValueError)):
        Terms(price, currency, latency, availability, Decimal("3600"))


def test_availability_window_is_positive_seconds():
    with pytest.raises(ValueError):
        Terms(Decimal("1"), "USD", Decimal("10"), Decimal("99"), Decimal("0"))


def test_federation_drives_real_opt_in_clients_and_preserves_run():
    class Client:
        def __init__(self, agent):
            self.agent = agent
            self.received = []

        async def run(self, prompt, **kwargs):
            return prompt

        async def negotiate(self, message):
            self.received.append(message)
            if self.agent == "seller":
                return Message(
                    "neg-1",
                    "msg-2",
                    "seller",
                    "buyer",
                    MessageKind.COUNTER,
                    terms("12.50", "150", "99.95"),
                    message.message_id,
                )
            return Message(
                "neg-1",
                "msg-3",
                "buyer",
                "seller",
                MessageKind.ACCEPT,
                message.terms,
                message.message_id,
            )

    buyer, seller = Client("buyer"), Client("seller")
    federation = AgentFederation()
    for name, client in [("buyer", buyer), ("seller", seller)]:
        federation.register_agent(AgentMetadata(id=name, model="test"), client=client)
    result = asyncio.run(federation.negotiate(offer(), max_rounds=3, deliver_terminal=False))
    assert result.status == NegotiationStatus.ACCEPTED
    assert result.agreement == terms("12.50", "150", "99.95")
    assert seller.received == [offer()]
    assert buyer.received == [result.messages[1]]
    assert asyncio.run(federation.run("hello", agent_id="buyer")) == "hello"


@pytest.mark.parametrize("terminal_kind", [MessageKind.ACCEPT, MessageKind.REJECT])
def test_bilateral_delivery_notifies_terminal_recipient(terminal_kind):
    class Client:
        def __init__(self, agent):
            self.agent = agent
            self.received = []

        async def run(self, prompt, **kwargs):
            return prompt

        async def negotiate(self, message):
            self.received.append(message)
            kind = MessageKind.COUNTER if self.agent == "seller" else terminal_kind
            return Message(
                message.negotiation_id,
                f"reply-{self.agent}",
                message.recipient,
                message.sender,
                kind,
                None if kind == MessageKind.REJECT else message.terms,
                message.message_id,
            )

        async def receive_negotiation(self, message):
            self.received.append(message)

    buyer, seller = Client("buyer"), Client("seller")
    federation = AgentFederation()
    for name, client in (("buyer", buyer), ("seller", seller)):
        federation.register_agent(AgentMetadata(id=name, model="test"), client=client)
    result = asyncio.run(federation.negotiate(offer(), max_rounds=2, deliver_terminal=True))
    assert seller.received == [result.messages[0], result.messages[2]]
    assert buyer.received == [result.messages[1]]
    assert result.messages[-1].kind == terminal_kind
    assert result.status == (
        NegotiationStatus.ACCEPTED
        if terminal_kind == MessageKind.ACCEPT
        else NegotiationStatus.REJECTED
    )


def test_bilateral_terminal_delivery_failure_propagates():
    class Client:
        async def run(self, prompt):
            return prompt

        def negotiate(self, message):
            return Message(
                message.negotiation_id,
                "end",
                message.recipient,
                message.sender,
                MessageKind.ACCEPT,
                message.terms,
                message.message_id,
            )

        def receive_negotiation(self, message):
            raise RuntimeError("delivery failed")

    federation = AgentFederation()
    for name in ("buyer", "seller"):
        federation.register_agent(AgentMetadata(id=name, model="test"), client=Client())
    with pytest.raises(RuntimeError, match="delivery failed"):
        asyncio.run(federation.negotiate(offer(), deliver_terminal=True))


def test_bilateral_delivery_requires_both_receivers_before_exchange():
    seen = []

    class Client:
        def run(self, prompt):
            return prompt

        def negotiate(self, message):
            seen.append(message)

    federation = AgentFederation()
    for name in ("buyer", "seller"):
        federation.register_agent(AgentMetadata(id=name, model="test"), client=Client())
    with pytest.raises(TypeError, match="receive_negotiation"):
        asyncio.run(federation.negotiate(offer(), deliver_terminal=True))
    assert not seen


@pytest.mark.parametrize("kind", [MessageKind.ACCEPT, MessageKind.REJECT])
def test_default_delivery_notifies_initial_sender(kind):
    class Client:
        def __init__(self):
            self.received = []

        def run(self, prompt):
            return prompt

        def negotiate(self, message):
            return Message(
                message.negotiation_id,
                "terminal",
                message.recipient,
                message.sender,
                kind,
                message.terms if kind == MessageKind.ACCEPT else None,
                message.message_id,
            )

        def receive_negotiation(self, message):
            self.received.append(message)

    federation = AgentFederation()
    buyer, seller = Client(), Client()
    federation.register_agent(AgentMetadata(id="buyer", model="test"), client=buyer)
    federation.register_agent(AgentMetadata(id="seller", model="test"), client=seller)
    result = asyncio.run(federation.negotiate(offer()))
    assert buyer.received == [result.messages[-1]]


def test_default_preflights_terminal_receivers_before_exchange():
    seen = []

    class Client:
        def run(self, prompt):
            return prompt

        def negotiate(self, message):
            seen.append(message)

    federation = AgentFederation()
    for name in ("buyer", "seller"):
        federation.register_agent(AgentMetadata(id=name, model="test"), client=Client())
    with pytest.raises(TypeError, match="receive_negotiation"):
        asyncio.run(federation.negotiate(offer()))
    assert not seen


def test_bounded_policy_overlap_and_concession():
    from synapsekit.agents.negotiation import BoundedPolicy, NegotiationRole

    buyer = BoundedPolicy(NegotiationRole.BUYER, terms("15", "200", "99.9"))
    seller = BoundedPolicy(NegotiationRole.SELLER, terms("12", "150", "99.95"))
    session = Negotiation(offer(), max_rounds=2)
    counter = seller.respond(session.messages[-1], message_id="counter", can_counter=True)
    assert counter.kind == MessageKind.COUNTER
    assert counter.terms == seller.reservation
    session.apply(counter)
    accepted = buyer.respond(counter, message_id="accept", can_counter=False)
    session.apply(accepted)
    assert session.status == NegotiationStatus.ACCEPTED
    assert buyer.accepts(session.agreement) and seller.accepts(session.agreement)


@pytest.mark.parametrize(
    "seller_reservation",
    [
        terms("16", "150", "99.95"),  # price
        terms("12", "250", "99.95"),  # latency
        terms("12", "150", "99.8"),  # availability
        terms("12", "150", "99.95", "EUR"),  # currency
        replace(terms("12", "150", "99.95"), availability_window_seconds=Decimal("7200")),
    ],
)
def test_bounded_policy_incompatible_reservations_reject_at_cap(seller_reservation):
    from synapsekit.agents.negotiation import BoundedPolicy, NegotiationRole

    buyer = BoundedPolicy(NegotiationRole.BUYER, terms("15", "200", "99.9"))
    seller = BoundedPolicy(NegotiationRole.SELLER, seller_reservation)
    initial = replace(offer(), terms=buyer.reservation)
    session = Negotiation(initial, max_rounds=1)
    rejected = seller.respond(initial, message_id="reject", can_counter=False)
    session.apply(rejected)
    assert session.status == NegotiationStatus.REJECTED
    assert session.agreement is None
    assert not seller.accepts(buyer.reservation)
    assert not buyer.accepts(seller.reservation)


def test_incompatible_policies_counter_then_reject_without_false_agreement():
    from synapsekit.agents.negotiation import BoundedPolicy, NegotiationRole

    buyer = BoundedPolicy(NegotiationRole.BUYER, terms("15", "200", "99.9"))
    seller = BoundedPolicy(NegotiationRole.SELLER, terms("16", "150", "99.95"))
    session = Negotiation(offer(), max_rounds=2)
    counter = seller.respond(offer(), message_id="counter", can_counter=True)
    session.apply(counter)
    rejection = buyer.respond(counter, message_id="reject", can_counter=False)
    session.apply(rejection)
    assert session.status == NegotiationStatus.REJECTED
    assert session.agreement is None
    assert rejection.kind == MessageKind.REJECT


def test_bounded_policy_rejects_currency_mismatch_even_when_counter_allowed():
    from synapsekit.agents.negotiation import BoundedPolicy, NegotiationRole

    policy = BoundedPolicy(NegotiationRole.SELLER, terms("12", "150", "99.95", "EUR"))
    response = policy.respond(offer(), message_id="reject", can_counter=True)
    assert response.kind == MessageKind.REJECT
    assert response.terms is None


def test_federation_requires_opt_in_and_registered_parties():
    federation = AgentFederation()
    federation.register_agent(AgentMetadata(id="seller", model="test"), client=lambda x: x)
    with pytest.raises(KeyError):
        asyncio.run(federation.negotiate(offer()))
    federation.register_agent(AgentMetadata(id="buyer", model="test"), client=lambda x: x)
    with pytest.raises(TypeError, match="negotiate"):
        asyncio.run(federation.negotiate(offer()))


def test_local_executor_can_reject_no_overlap_without_execution():
    class Executor:
        def __call__(self, prompt):
            raise AssertionError("negotiation must not run the task")

        def negotiate(self, message):
            return Message(
                message.negotiation_id,
                "no-deal",
                message.recipient,
                message.sender,
                MessageKind.REJECT,
                None,
                message.message_id,
            )

    federation = AgentFederation()
    for name in ("buyer", "seller"):
        federation.register_agent(AgentMetadata(id=name, model="test"), client=Executor())
    result = asyncio.run(federation.negotiate(offer(), deliver_terminal=False))
    assert result.status == NegotiationStatus.REJECTED
    assert result.agreement is None


def test_federation_rejects_counter_after_round_limit():
    class Client:
        async def run(self, prompt):
            return prompt

        async def negotiate(self, message):
            return Message(
                message.negotiation_id,
                "counter",
                message.recipient,
                message.sender,
                MessageKind.COUNTER,
                message.terms,
                message.message_id,
            )

    federation = AgentFederation()
    for name in ("buyer", "seller"):
        federation.register_agent(AgentMetadata(id=name, model="test"), client=Client())
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(federation.negotiate(offer(), max_rounds=1, deliver_terminal=False))
