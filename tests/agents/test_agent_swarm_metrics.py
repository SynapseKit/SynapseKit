"""AgentSwarm drives the PrometheusMetrics dashboard signals end-to-end (#734).

Uses a hand-written metrics collector (no MagicMock) to prove that running an
auction emits the three signals the #734 live dashboard needs: bids (bids/sec),
wins per agent (win rate per agent), and reward (avg reward).
"""

from synapsekit import AgentSwarm, Bid, MarketPolicy
from synapsekit.agents import AgentMetadata, InMemoryAgentRegistry


class RecordingMetrics:
    """Records swarm metric calls the way PrometheusMetrics would consume them."""

    def __init__(self) -> None:
        self.bids: list[dict] = []
        self.auctions: list[dict] = []
        self.wins: list[dict] = []

    def record_swarm_bid(self, **kwargs) -> None:
        self.bids.append(kwargs)

    def record_swarm_auction(self, **kwargs) -> None:
        self.auctions.append(kwargs)

    def record_swarm_win(self, **kwargs) -> None:
        self.wins.append(kwargs)


class MarketClient:
    def __init__(self, agent_id: str, *, cost: float, quality: float) -> None:
        self.agent_id = agent_id
        self.cost = cost
        self.quality = quality

    def bid(self, task: str, **kwargs):
        return Bid(
            agent_id=self.agent_id,
            estimated_cost=self.cost,
            estimated_quality=self.quality,
            confidence=0.9,
            task_category=kwargs["task_category"],
        )

    async def run(self, prompt: str, **kwargs):
        return {"agent_id": self.agent_id, "actual_quality": self.quality, "actual_cost": self.cost}


async def test_swarm_emits_bid_win_and_reward_signals():
    metrics = RecordingMetrics()
    swarm = AgentSwarm(
        market=MarketPolicy(budget_per_task=100, seed=42, exploration_rate=0),
        registry=InMemoryAgentRegistry(),
        metrics=metrics,
    )
    swarm.register_agent(
        AgentMetadata(id="researcher", model="mock", capacity=2),
        client=MarketClient("researcher", cost=30.0, quality=0.96),
    )
    swarm.register_agent(
        AgentMetadata(id="summarizer", model="mock", capacity=2),
        client=MarketClient("summarizer", cost=8.0, quality=0.74),
    )

    await swarm.execute("Write a market brief", task_category="research")

    # bids/sec — one bid signal per bidding agent, labelled by agent + category.
    assert {b["agent_id"] for b in metrics.bids} == {"researcher", "summarizer"}
    assert all(b["task_category"] == "research" for b in metrics.bids)

    # auction — bid count for the auction latency/throughput panels.
    assert len(metrics.auctions) == 1
    assert metrics.auctions[0]["bid_count"] == 2

    # win rate per agent + avg reward — one win signal for the winner, carrying
    # the reward and settlement cost the dashboard aggregates.
    assert len(metrics.wins) == 1
    win = metrics.wins[0]
    assert win["agent_id"] == "researcher"
    assert isinstance(win["reward"], float)
    assert win["settlement_cost"] is not None
