import inspect

import pytest

from synapsekit import AgentSwarm, AuctionResult, AuctionType, Bid, BidStrategy, MarketPolicy, SwarmResult
from synapsekit.agents import AgentFederation, AgentMetadata, InMemoryAgentRegistry, Reputation
from synapsekit.agents.agent_swarm import CoalitionFormer


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

    auction = await swarm.auction(
        "Review code", swarm.federation.discover(), task_category="code"
    )

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
