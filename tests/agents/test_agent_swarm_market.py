import asyncio

import pytest

from synapsekit import AgentSwarm, AuctionType, Bid, BidStrategy, MarketPolicy
from synapsekit.agents import AgentFederation, AgentMetadata, InMemoryAgentRegistry, Reputation


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


@pytest.mark.asyncio
async def test_sealed_bid_market_selects_best_cost_quality_tradeoff():
    swarm, clients = make_swarm()

    result = await swarm.execute("Write a market brief", task_category="research")

    assert result.winners == ["researcher"]
    assert clients["researcher"].calls == ["Write a market brief"]
    assert clients["summarizer"].calls == []
    assert result.auction.auction_type == AuctionType.SEALED_BID
    assert swarm.reputation.get("researcher", "research").wins == 1
    assert "winner: researcher" in swarm.trace_to_mermaid()


@pytest.mark.asyncio
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

    assert result.winners == ["researcher", "summarizer"]
    assert clients["researcher"].calls
    assert clients["summarizer"].calls
    assert result.output["mode"] == "multi_winner"


@pytest.mark.asyncio
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

    assert len(result.winners) == 2
    assert result.auction.coalition is True
    assert result.output["mode"] == "coalition"


@pytest.mark.asyncio
async def test_vickrey_uses_second_best_bid_for_settlement():
    policy = MarketPolicy(
        auction_type="vickrey",
        budget_per_task=100,
        seed=42,
        exploration_rate=0,
    )
    swarm, _ = make_swarm(policy=policy)

    result = await swarm.execute("Analyze", task_category="research")

    assert result.winners == ["researcher"]
    assert result.auction.settlement_cost == 8.0


@pytest.mark.asyncio
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

    assert result.winners == ["strong"]
    assert result.output["agent_id"] == "strong"


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


def test_synthetic_bids_use_reputation_when_agent_has_no_bidder():
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

    auction = asyncio.run(swarm.auction("Review code", swarm.federation.discover(), task_category="code"))

    assert auction.bids[0].estimated_quality == 0.9
    assert auction.bids[0].estimated_cost == 3.0
