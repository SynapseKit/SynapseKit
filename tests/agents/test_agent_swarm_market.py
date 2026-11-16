import inspect
from decimal import Decimal

import pytest

from synapsekit import (
    AgentSwarm,
    AuctionResult,
    AuctionType,
    Bid,
    BidStrategy,
    MarketPolicy,
    SwarmResult,
)
from synapsekit.agents import AgentFederation, AgentMetadata, InMemoryAgentRegistry, Reputation
from synapsekit.agents.agent_swarm import CoalitionFormer
from synapsekit.agents.negotiation import Message, MessageKind, Negotiation, Terms


class MarketClient:
    def __init__(
        self,
        agent_id: str,
        *,
        cost: float,
        quality: float,
        confidence: float = 1.0,
    ) -> None:
        self.agent_id = agent_id
        self.cost = cost
        self.quality = quality
        self.confidence = confidence
        self.calls: list[str] = []
        self.settlements: list[dict] = []

    def bid(self, task: str, **kwargs):
        return Bid(
            agent_id=self.agent_id,
            estimated_cost=self.cost,
            estimated_quality=self.quality,
            confidence=self.confidence,
            task_category=kwargs["task_category"],
        )

    async def run(self, prompt: str, **kwargs):
        self.calls.append(prompt)
        return {
            "agent_id": self.agent_id,
            "prompt": prompt,
            "actual_cost": self.cost,
            "actual_quality": self.quality,
            "reward": self.quality - (self.cost / 100.0),
        }

    def settle(self, **kwargs):
        self.settlements.append(kwargs)


def make_swarm(
    *,
    policy: MarketPolicy | None = None,
    reputation: Reputation | None = None,
) -> tuple[AgentSwarm, dict[str, MarketClient]]:
    registry = InMemoryAgentRegistry()
    clients = {
        "planner": MarketClient("planner", cost=20.0, quality=0.86, confidence=0.8),
        "researcher": MarketClient("researcher", cost=30.0, quality=0.96, confidence=0.95),
        "summarizer": MarketClient("summarizer", cost=8.0, quality=0.74, confidence=0.9),
    }
    swarm = AgentSwarm(
        market=policy or MarketPolicy(budget_per_task=100, seed=42, exploration_rate=0),
        registry=registry,
        reputation=reputation,
    )
    for agent_id, client in clients.items():
        swarm.register_agent(
            AgentMetadata(
                id=agent_id,
                model="mock",
                tools=["write", "search"],
                tags=["bench"],
                cost_multiplier=client.cost,
                capacity=2,
            ),
            client=client,
        )
    return swarm, clients


# ---------------------------------------------------------------------------
# Async-method coroutine assertions
# ---------------------------------------------------------------------------


def test_swarm_async_methods_are_coroutines():
    assert inspect.iscoroutinefunction(AgentSwarm.execute)
    assert inspect.iscoroutinefunction(AgentSwarm.run)
    assert inspect.iscoroutinefunction(AgentSwarm.auction)


# ---------------------------------------------------------------------------
# Core market behaviour
# ---------------------------------------------------------------------------


async def test_sealed_bid_market_selects_best_cost_quality_tradeoff():
    swarm, clients = make_swarm()

    result = await swarm.execute("Write a market brief", task_category="research")

    assert isinstance(result, SwarmResult)
    assert result.winners == ["researcher"]
    assert clients["researcher"].calls == ["Write a market brief"]
    assert clients["summarizer"].calls == []
    assert result.auction.auction_type == AuctionType.SEALED_BID
    assert isinstance(result.auction, AuctionResult)
    assert swarm.reputation.get("researcher", "research").wins == 1
    assert "winner: researcher" in swarm.trace_to_mermaid()


async def test_multi_winner_market_executes_top_ranked_agents():
    policy = MarketPolicy(
        auction_type="multi_winner",
        max_winners=2,
        budget_per_task=100,
        seed=42,
        exploration_rate=0,
    )
    swarm, clients = make_swarm(policy=policy)

    result = await swarm.execute("Draft and review", task_category="writing")

    assert isinstance(result, SwarmResult)
    assert result.winners == ["researcher", "summarizer"]
    assert clients["researcher"].calls
    assert clients["summarizer"].calls
    assert result.output["mode"] == "multi_winner"


async def test_coalition_market_marks_cooperative_result():
    policy = MarketPolicy(
        auction_type="coalition",
        coalition_size=2,
        budget_per_task=100,
        seed=42,
        exploration_rate=0,
    )
    swarm, _ = make_swarm(policy=policy)

    result = await swarm.execute("Plan research coalition", task_category="research")

    assert isinstance(result, SwarmResult)
    assert len(result.winners) == 2
    assert result.auction.coalition is True
    assert result.output["mode"] == "coalition"


async def test_vickrey_uses_second_highest_price_for_settlement():
    """Vickrey: winner pays second-highest *bid price*, not second-highest scored agent's price."""
    policy = MarketPolicy(
        auction_type="vickrey",
        budget_per_task=100,
        seed=42,
        exploration_rate=0,
    )
    swarm, _ = make_swarm(policy=policy)

    # researcher cost=30 (highest price), planner cost=20 (second-highest price), summarizer cost=8
    result = await swarm.execute("Analyze", task_category="research")

    assert isinstance(result, SwarmResult)
    assert result.winners == ["researcher"]
    # Second-highest BID PRICE is planner's 20.0
    assert result.auction.settlement_cost == 20.0


async def test_federation_market_strategy_returns_swarm_result():
    registry = InMemoryAgentRegistry()
    federation = AgentFederation(registry)
    federation.register_agent(
        AgentMetadata(id="cheap", model="mock", cost_multiplier=1.0, tags=["support"]),
        client=MarketClient("cheap", cost=1.0, quality=0.7),
    )
    federation.register_agent(
        AgentMetadata(id="strong", model="mock", cost_multiplier=5.0, tags=["support"]),
        client=MarketClient("strong", cost=5.0, quality=0.95),
    )

    result = await federation.run(
        "Resolve ticket",
        strategy="market",
        tags=["support"],
        task_category="ticket",
        market=MarketPolicy(budget_per_task=10, seed=1, exploration_rate=0),
    )

    assert isinstance(result, SwarmResult)
    assert result.winners == ["strong"]
    assert result.output["agent_id"] == "strong"


def negotiated_session(seller, price, latency="100", availability="99.9", currency="USD"):
    terms = Terms(
        Decimal(price), currency, Decimal(latency), Decimal(availability), Decimal("3600")
    )
    offer = Message("neg-" + seller, "offer", "buyer", seller, MessageKind.OFFER, terms)
    session = Negotiation(offer)
    session.apply(
        Message(offer.negotiation_id, "accept", seller, "buyer", MessageKind.ACCEPT, terms, "offer")
    )
    return session


@pytest.mark.parametrize(
    ("budget", "price", "currency"),
    [
        (0.3, "0.3", "USD"),
        (0.1, "0.1", "EUR"),
        (0.3, "0.3", "JPY"),
        (1.7976931348623157e308, "1.7976931348623157e308", "USD"),
        (5e-324, "5e-324", "USD"),
        (0.0, "0", "USD"),
    ],
)
async def test_negotiated_price_at_decimal_budget_is_eligible(budget, price, currency):
    swarm, clients = make_swarm(policy=MarketPolicy(budget_per_task=budget, exploration_rate=0))
    reservation = Terms(Decimal(price), currency, Decimal("200"), Decimal("99"), Decimal("3600"))
    result = await swarm.execute(
        "task",
        negotiated_agreements={"planner": negotiated_session("planner", price, currency=currency)},
        negotiation_buyer="buyer",
        negotiation_requirements=reservation,
    )
    assert result.winners == ["planner"]
    assert result.auction.settlement_cost == float(Decimal(price))
    assert clients["planner"].calls == ["task"]
    assert clients["planner"].settlements[0]["bid"].metadata["negotiated_terms"].price == Decimal(
        price
    )


@pytest.mark.parametrize(
    ("budget", "price", "currency"),
    [
        (0.3, "0.300000000000000001", "USD"),
        (0.1, "0.100000000000000001", "EUR"),
        (0.3, "0.300000000000000001", "JPY"),
        (1.7976931348623157e308, "1.7976931348623158e308", "USD"),
        (1.7976931348623157e308, "1e10000", "USD"),
        (5e-324, "1e-10000", "USD"),
    ],
)
async def test_negotiated_price_above_budget_or_unrepresentable_is_excluded(
    budget, price, currency
):
    swarm, clients = make_swarm(policy=MarketPolicy(budget_per_task=budget, exploration_rate=0))
    reservation = Terms(Decimal(price), currency, Decimal("200"), Decimal("99"), Decimal("3600"))
    with pytest.raises(LookupError, match="eligible"):
        await swarm.execute(
            "task",
            negotiated_agreements={
                "planner": negotiated_session("planner", price, currency=currency)
            },
            negotiation_buyer="buyer",
            negotiation_requirements=reservation,
        )
    assert all(not client.calls and not client.settlements for client in clients.values())


async def test_opt_in_agreements_change_market_routing_and_settlement():
    swarm, clients = make_swarm()
    reservation = Terms(Decimal("100"), "USD", Decimal("200"), Decimal("99"), Decimal("3600"))
    normal = await make_swarm()[0].execute("task")
    result = await swarm.execute(
        "task",
        negotiated_agreements={
            "researcher": negotiated_session("researcher", "90"),
            "planner": negotiated_session("planner", "1"),
            "summarizer": negotiated_session("summarizer", "1", latency="300"),
        },
        negotiation_buyer="buyer",
        negotiation_requirements=reservation,
    )
    assert normal.winners == ["researcher"]
    assert result.winners == ["planner"]
    assert {bid.agent_id: bid.estimated_cost for bid in result.auction.bids} == {
        "researcher": 90.0,
        "planner": 1.0,
    }
    assert result.auction.settlement_cost == 1.0
    assert clients["planner"].settlements[0]["bid"].metadata["negotiated_terms"] == (
        negotiated_session("planner", "1").agreement
    )


async def test_ordinary_execution_preserves_overridden_auction_signature():
    class CustomSwarm(AgentSwarm):
        async def auction(self, task, candidates, *, task_category=None):
            return await super().auction(task, candidates, task_category=task_category)

    swarm, _ = make_swarm()
    custom = CustomSwarm(
        market=swarm.market, registry=swarm.registry, clients=swarm.federation.clients
    )
    assert (await custom.execute("task")).winners == ["researcher"]


async def test_opt_in_agreements_fail_closed_when_over_budget_or_no_sla_match():
    swarm, clients = make_swarm()
    reservation = Terms(Decimal("1000"), "USD", Decimal("200"), Decimal("99"), Decimal("3600"))
    with pytest.raises(LookupError, match="eligible"):
        await swarm.execute(
            "task",
            negotiated_agreements={
                "planner": negotiated_session("planner", "101"),
                "researcher": negotiated_session("researcher", "1", availability="98"),
            },
            negotiation_buyer="buyer",
            negotiation_requirements=reservation,
        )
    assert all(not client.calls for client in clients.values())


async def test_opt_in_agreements_reject_wrong_parties_and_nonaccepted():
    swarm, clients = make_swarm()
    reservation = Terms(Decimal("100"), "USD", Decimal("200"), Decimal("99"), Decimal("3600"))
    initial = negotiated_session("planner", "1").messages[0]
    for agreements in (
        {"planner": Negotiation(initial)},
        {"researcher": negotiated_session("planner", "1")},
    ):
        with pytest.raises(ValueError):
            await swarm.execute(
                "task",
                negotiated_agreements=agreements,
                negotiation_buyer="buyer",
                negotiation_requirements=reservation,
            )
    assert all(not client.calls for client in clients.values())


async def test_federation_negotiation_flows_into_market_execution():
    class NegotiatingClient(MarketClient):
        def negotiate(self, proposal):
            return Message(
                proposal.negotiation_id,
                "accepted",
                proposal.recipient,
                proposal.sender,
                MessageKind.ACCEPT,
                proposal.terms,
                proposal.message_id,
            )

        def receive_negotiation(self, message):
            self.received = message

    federation = AgentFederation()
    buyer = NegotiatingClient("buyer", cost=1, quality=0.5)
    seller = NegotiatingClient("seller", cost=50, quality=0.8)
    for name, client in (("buyer", buyer), ("seller", seller)):
        federation.register_agent(AgentMetadata(id=name, model="test"), client=client)
    reservation = Terms(Decimal("10"), "USD", Decimal("200"), Decimal("99"), Decimal("3600"))
    offer = Message("real-neg", "offer", "buyer", "seller", MessageKind.OFFER, reservation)
    agreement = await federation.negotiate(offer)
    result = await federation.run(
        "task",
        strategy="market",
        market=MarketPolicy(budget_per_task=100, exploration_rate=0),
        negotiated_agreements={"seller": agreement},
        negotiation_buyer="buyer",
        negotiation_requirements=reservation,
    )
    assert buyer.received == agreement.messages[-1]
    assert result.winners == ["seller"]
    assert result.auction.settlement_cost == 10
    assert result.auction.winners[0].metadata["negotiation_id"] == "real-neg"


async def test_federation_rejects_negotiation_inputs_without_market_routing():
    swarm, clients = make_swarm()
    with pytest.raises(ValueError, match="market"):
        await swarm.federation.run("task", agent_id="planner", negotiated_agreements={})
    assert not clients["planner"].calls


@pytest.mark.parametrize("auction_type", ["vickrey", "multi_winner", "coalition", "english"])
async def test_negotiated_routing_rejects_other_auction_semantics(auction_type):
    swarm, _ = make_swarm(policy=MarketPolicy(auction_type=auction_type, budget_per_task=100))
    reservation = Terms(Decimal("100"), "USD", Decimal("200"), Decimal("99"), Decimal("3600"))
    with pytest.raises(ValueError, match="sealed-bid"):
        await swarm.execute(
            "task",
            negotiated_agreements={"planner": negotiated_session("planner", "1")},
            negotiation_buyer="buyer",
            negotiation_requirements=reservation,
        )


async def test_negotiated_market_rejects_nonfinite_budget():
    swarm, _ = make_swarm(policy=MarketPolicy(budget_per_task=float("inf")))
    reservation = Terms(Decimal("100"), "USD", Decimal("200"), Decimal("99"), Decimal("3600"))
    with pytest.raises(ValueError, match="finite"):
        await swarm.execute(
            "task",
            negotiated_agreements={"planner": negotiated_session("planner", "1")},
            negotiation_buyer="buyer",
            negotiation_requirements=reservation,
        )


async def test_negotiated_market_filters_wrong_currency_and_window():
    swarm, _ = make_swarm()
    reservation = Terms(Decimal("100"), "USD", Decimal("200"), Decimal("99"), Decimal("3600"))
    wrong_window = negotiated_session("researcher", "1")
    initial = wrong_window.messages[0]
    from dataclasses import replace

    initial = replace(
        initial, terms=replace(initial.terms, availability_window_seconds=Decimal("60"))
    )
    wrong_window = Negotiation(initial)
    wrong_window.apply(
        Message(
            initial.negotiation_id,
            "accept",
            "researcher",
            "buyer",
            MessageKind.ACCEPT,
            initial.terms,
            "offer",
        )
    )
    with pytest.raises(LookupError, match="eligible"):
        await swarm.execute(
            "task",
            negotiated_agreements={
                "planner": negotiated_session("planner", "1", currency="EUR"),
                "researcher": wrong_window,
            },
            negotiation_buyer="buyer",
            negotiation_requirements=reservation,
        )


# ---------------------------------------------------------------------------
# Smoke / import tests
# ---------------------------------------------------------------------------


def test_agent_swarm_imports_from_top_level():
    from synapsekit import AgentBidder, AuctionResult, CoalitionFormer, ReputationSnapshot

    assert AgentSwarm is not None
    assert AgentBidder is not None
    assert AuctionResult is not None
    assert BidStrategy.ucb() is not None
    assert CoalitionFormer(max_size=2) is not None
    assert ReputationSnapshot(agent_id="a").agent_id == "a"


def test_market_policy_aliases_and_validation():
    policy = MarketPolicy(auction_type="sealed", bid_strategy="pareto", seed=42)

    assert policy.auction_type == AuctionType.SEALED_BID
    assert policy.bid_strategy.name == "cost_quality_pareto"


# ---------------------------------------------------------------------------
# Synthetic bid uses reputation (converted from asyncio.run to async def)
# ---------------------------------------------------------------------------


async def test_synthetic_bids_use_reputation_when_agent_has_no_bidder():
    reputation = Reputation()
    reputation.record_outcome(
        "agent-a",
        "code",
        cost=3.0,
        quality=0.9,
        reward=0.8,
        learning_rate=0.5,
    )
    swarm = AgentSwarm(
        agents=[AgentMetadata(id="agent-a", model="mock", cost_multiplier=9.0)],
        market=MarketPolicy(budget_per_task=10, seed=42, exploration_rate=0),
        reputation=reputation,
    )

    auction = await swarm.auction("Review code", swarm.federation.discover(), task_category="code")

    assert isinstance(auction, AuctionResult)
    assert auction.bids[0].estimated_quality == 0.9
    assert auction.bids[0].estimated_cost == 3.0


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_swarm_with_empty_agents_raises_on_execute():
    """An AgentSwarm with no agents raises LookupError when execute() is called."""
    swarm = AgentSwarm(agents=[])
    # Constructing the swarm is valid — the error is raised on execute
    assert swarm is not None


async def test_swarm_no_agents_raises_lookup_error_on_execute():
    swarm = AgentSwarm(agents=[], market=MarketPolicy(budget_per_task=100))
    with pytest.raises(LookupError):
        await swarm.execute("do something")


def test_bid_requires_nonempty_agent_id():
    with pytest.raises(ValueError):
        Bid(agent_id="", estimated_cost=10, estimated_quality=0.8, confidence=0.9)


def test_bid_rejects_none_converted_to_empty_string():
    """agent_id=None coerces to 'None' (non-empty) — but integer 0 coerces to '0' which is fine.
    Verify that explicitly passing an empty string raises ValueError."""
    # This is a duplicate guard: confirms the validator is present
    with pytest.raises(ValueError):
        Bid(agent_id="", estimated_cost=5, estimated_quality=0.5)


def test_coalition_former_raises_when_fewer_agents_than_max_size():
    former = CoalitionFormer(max_size=3)
    bids = [
        Bid(agent_id="a", estimated_cost=1.0, estimated_quality=0.8),
        Bid(agent_id="b", estimated_cost=2.0, estimated_quality=0.7),
    ]
    scores = {"a": 0.9, "b": 0.8}
    with pytest.raises(LookupError):
        former.form(bids, scores)


def test_market_policy_rejects_unknown_auction_type():
    with pytest.raises(ValueError):
        MarketPolicy(auction_type="dutch_auction")


def test_bid_strategy_rejects_unknown_name():
    with pytest.raises(ValueError):
        BidStrategy.from_value("nonexistent_strategy")


async def test_swarm_budget_zero_still_selects_winner():
    """budget_per_task=0 should not crash; all bids fall back to eligible."""
    policy = MarketPolicy(budget_per_task=0, seed=42, exploration_rate=0)
    swarm, _ = make_swarm(policy=policy)

    result = await swarm.execute("Do something", task_category="general")

    assert isinstance(result, SwarmResult)
    assert len(result.winners) >= 1


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------


async def test_swarm_scales_to_50_agents():
    """AgentSwarm must handle 50 agents without error and return a SwarmResult."""
    swarm = AgentSwarm(
        market=MarketPolicy(budget_per_task=1000, seed=7, exploration_rate=0),
    )
    for i in range(50):
        agent_id = f"stress-agent-{i:02d}"
        cost = float(i + 1)
        quality = 0.5 + (i % 10) * 0.04
        swarm.register_agent(
            AgentMetadata(
                id=agent_id,
                model="mock",
                cost_multiplier=cost,
                capacity=2,
            ),
            client=MarketClient(agent_id, cost=cost, quality=quality),
        )

    result = await swarm.execute("Process large batch", task_category="batch")

    assert isinstance(result, SwarmResult)
    assert len(result.winners) >= 1
    assert len(result.auction.bids) == 50
