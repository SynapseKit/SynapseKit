# Agent price and SLA negotiation

Negotiation is an **opt-in, bilateral** protocol, separate from normal routing.
An additional opt-in bridge to sealed-bid market routing is described below.
Both parties must already be registered, healthy, and have a
client (or `LocalAgentClient` executor) with `negotiate(message)`. The handler
may be synchronous or asynchronous and must return a `Message`. The federation
passes each proposal to its recipient; it never synthesizes a bid or acceptance.

For bilateral terminal delivery (the default), implement
`receive_negotiation(message)` on **both** clients. Before exchanging proposals,
the federation checks for both handlers. Once an ACCEPT or REJECT has been
validated, the recipient's receiver is called and awaited if necessary; its return
value is ignored. A receiver failure propagates and the federation does not return
a completed session. Delivery may nevertheless have had side effects before a
failure, so the caller must reconcile/retry using its own idempotency mechanism.
This is not durable or exactly-once delivery. Explicitly setting
`deliver_terminal=False` selects transcript-only mode: the terminal message is
recorded in the returned transcript but is **not** delivered to its recipient.

`BoundedPolicy` provides a simple deterministic reservation strategy. A buyer's
reservation gives its maximum price, maximum latency and minimum availability; a
seller's gives its minimum price, minimum latency it can meet and maximum
availability it can provide. A proposal is accepted only if all the receiving
party's bounds are met and currency and availability window match exactly.
Otherwise, the policy counters with its own reservation terms (a one-step
concession) if a counter is permitted; it rejects at the round limit or when the
currency/window differs. The caller must ensure its **initial** offer meets its
own reservation, allocate unique message IDs, and track the proposal count to
supply `can_counter` accurately. Policies cannot establish the other party's
bounds; they rely on that party to enforce them.

```python
import asyncio
from decimal import Decimal

from synapsekit.agents.agent_registry import AgentMetadata
from synapsekit.agents.federation import AgentFederation
from synapsekit.agents.negotiation import (
    BoundedPolicy, Message, MessageKind, NegotiationRole, Terms,
)


def terms(price, latency, availability):
    return Terms(Decimal(price), "USD", Decimal(latency),
                 Decimal(availability), Decimal("3600"))


class Client:
    def __init__(self, policy):
        self.policy = policy
        self.received = []

    async def run(self, prompt, **kwargs):
        return prompt  # Normal execution remains separate.

    async def negotiate(self, proposal):
        self.received.append(proposal)
        # This two-proposal example uses m1 only for the first proposal.
        # A real client must track the proposal count per negotiation.
        return self.policy.respond(proposal, message_id=proposal.message_id + "r",
                                   can_counter=proposal.message_id == "m1")

    async def receive_negotiation(self, message):
        self.received.append(message)  # Persist or acknowledge as appropriate.


async def main():
    buyer = Client(BoundedPolicy(NegotiationRole.BUYER, terms("15", "200", "99.9")))
    seller = Client(BoundedPolicy(NegotiationRole.SELLER, terms("12", "150", "99.95")))
    federation = AgentFederation()
    for name, client in (("buyer", buyer), ("seller", seller)):
        federation.register_agent(AgentMetadata(id=name, model="example"), client=client)
    initial = Message("neg-1", "m1", "buyer", "seller", MessageKind.OFFER,
                      terms("10", "200", "99.9"))
    assert buyer.policy.accepts(initial.terms)
    result = await federation.negotiate(initial, max_rounds=2, deliver_terminal=True)
    assert result.agreement == seller.policy.reservation
    assert buyer.policy.accepts(result.agreement) and seller.policy.accepts(result.agreement)
    assert seller.received[-1].kind == MessageKind.ACCEPT
    print(result.status, result.agreement, result.messages)

asyncio.run(main())
```

`Terms` uses finite `Decimal` values, not floats: nonnegative price **per task**,
three uppercase ASCII currency letters (syntax only, not an ISO currency registry),
positive maximum task latency in milliseconds, and minimum availability as a
percentage from 0 to 100 over a positive window in seconds. The application must
define how task latency, availability, and the window are measured. Currency is
fixed for the negotiation. A counter replaces all terms; acceptance must exactly
match the latest terms. Each response reverses sender/recipient, has a unique
message ID within the negotiation, and references the immediately previous
proposal. `Negotiation` can also be used directly with `apply(message)` and
exposes an immutable transcript snapshot, status and agreement. Concurrent calls
on the same session serialize validation and mutation; competing responses to the
same proposal cannot both succeed. Each property read is synchronized separately,
so multiple property reads are not a single combined snapshot. No lock is held
across asynchronous client calls.

`max_rounds` counts proposals, including the first offer (1–1000). At the limit,
the recipient may accept or reject; another counter raises `ValueError`. A
rejection ends without an agreement. Invalid messages, replayed IDs, stale
references, altered acceptances and post-terminal messages raise without changing
the transcript. A failed federation exchange propagates the exception; it does
not return a partial session. Handler exceptions propagate too. Async callers
can apply `asyncio.wait_for` for an overall timeout; synchronous handlers must
not block the event loop.

## Opt-in market routing

An accepted transcript can be used explicitly in a sealed-bid market execution:

```python
result = await federation.run(
    "Perform the task", strategy="market", market=MarketPolicy(budget_per_task=100),
    negotiated_agreements={"seller": result_from_negotiate},
    negotiation_buyer="buyer",
    negotiation_requirements=buyer_reservation_terms,
)
```

`MarketPolicy` is imported from `synapsekit.agents.agent_swarm`. The same three
keyword arguments work on `AgentSwarm.execute`. Only sellers with supplied,
accepted buyer-to-seller transcripts can compete. Each agreement must match the
specified parties. Agreements outside the buyer reservation (including currency,
latency, availability or window) or market budget are excluded. Existing
discovery and bidder eligibility still apply; if no eligible bid remains, execution
fails instead of falling back to an unnegotiated agent. Only a finite budget and
sealed-bid auction are supported in this mode; other auction semantics are rejected.

For negotiated budget checks, the already-normalized `MarketPolicy` float budget
is interpreted as the Decimal of its shortest round-trippable string (for example,
`0.3` means `Decimal("0.3")`). The agreed Decimal price must not exceed that
value, even if a slightly higher price would round to the same float. This applies
in any matching currency; the market does not perform currency conversion. Prices
that convert to infinity or nonzero prices that underflow to zero are excluded.

The accepted price replaces the bidder's estimated cost for scoring and the
sealed-bid settlement estimate; the original `Terms` and negotiation ID are
available in the winning `Bid.metadata`. The market converts Decimal prices to
floating-point values, so those estimates may be rounded. Agent-reported
`actual_cost` can differ from that estimate, and existing settlement callbacks
still receive that reported cost. Neither the callback nor this routing path pays
anyone or verifies latency or availability. Bid quality and confidence remain
the bidder's estimates; market exploration may affect selection.

Transcripts alone do not prove terminal delivery, authentication, task scope,
freshness, or exclusive use. The caller is responsible for verifying these before
supplying the per-execution mapping, including when it was obtained through
transcript-only mode or constructed locally. Without the opt-in inputs, ordinary
market routing is unchanged.

The transcript and replay protection are **in memory and per session**, not
persistent or cross-session; identifiers are not authentication or signatures.
The application must authenticate clients, protect any transport, allocate unique
negotiation/message IDs, and store records if needed. This is not a payment,
contract execution, SLA measurement, enforcement, or dispute system.
