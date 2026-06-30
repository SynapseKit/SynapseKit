"""Market-based AgentSwarm example.

Run with:
    python examples/agent_swarm_market.py
"""

from __future__ import annotations

import asyncio

from synapsekit import AgentMetadata, AgentSwarm, Bid, BidStrategy, MarketPolicy


class DemoAgent:
    def __init__(self, agent_id: str, cost: float, quality: float) -> None:
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
        return {
            "agent_id": self.agent_id,
            "answer": f"{self.agent_id} handled: {prompt}",
            "actual_cost": self.cost,
            "actual_quality": self.quality,
            "reward": self.quality - (self.cost / 100.0),
        }


async def main() -> None:
    swarm = AgentSwarm(
        market=MarketPolicy(
            bid_strategy=BidStrategy.cost_quality_pareto(),
            auction_type="sealed_bid",
            budget_per_task=100,
            seed=42,
            exploration_rate=0,
        )
    )

    agents = {
        "planner": DemoAgent("planner", cost=18, quality=0.84),
        "researcher": DemoAgent("researcher", cost=30, quality=0.96),
        "summarizer": DemoAgent("summarizer", cost=8, quality=0.74),
        "critic": DemoAgent("critic", cost=15, quality=0.88),
        "coder": DemoAgent("coder", cost=22, quality=0.90),
    }

    for agent_id, client in agents.items():
        swarm.register_agent(
            AgentMetadata(
                id=agent_id,
                model="demo",
                tools=["research", "write"],
                tags=["demo"],
                cost_multiplier=client.cost,
                capacity=2,
            ),
            client=client,
        )

    result = await swarm.execute(
        "Write a market analysis on quantum compute startups",
        task_category="research",
    )

    print("Winner:", result.winners[0])
    print("Output:", result.output["answer"])
    print("Reputation:", swarm.reputation.get(result.winners[0], "research").to_dict())
    print("\nTrace:")
    print(swarm.trace_to_mermaid())


if __name__ == "__main__":
    asyncio.run(main())
