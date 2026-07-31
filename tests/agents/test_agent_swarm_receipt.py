"""Spec test for the replayable role-allocation receipt requested on #734 by clementineCU.

Every auction should emit a receipt that can be replayed without trusting the
final answer: task_id, candidate agents, bid inputs each agent was allowed to
see, cost/quality prior version, bid value, selected role, rejected roles,
budget consumed, outcome score source, and the rule that updates future bids.

Implemented via the ``GroundedSignal`` provenance primitive (#822): the receipt
now carries the reputation prior each bid was scored against, the budget
allocated vs. consumed, and whether the outcome score was externally supplied
or self-reported by the winning agent.
"""

from synapsekit import AgentSwarm, Bid, GroundedSignal, MarketPolicy, SignalSource
from synapsekit.agents import AgentMetadata, InMemoryAgentRegistry


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
        # No actual_cost/actual_quality/reward in the output — forces AgentSwarm
        # to fall back to self-reported bid values when settling the outcome.
        return {"agent_id": self.agent_id, "prompt": prompt}


async def test_auction_receipt_is_replayable_without_trusting_final_answer():
    registry = InMemoryAgentRegistry()
    swarm = AgentSwarm(
        market=MarketPolicy(budget_per_task=100, seed=42, exploration_rate=0),
        registry=registry,
    )
    swarm.register_agent(
        AgentMetadata(id="researcher", model="mock", cost_multiplier=30.0, capacity=2),
        client=MarketClient("researcher", cost=30.0, quality=0.96),
    )
    swarm.register_agent(
        AgentMetadata(id="summarizer", model="mock", cost_multiplier=8.0, capacity=2),
        client=MarketClient("summarizer", cost=8.0, quality=0.74),
    )

    await swarm.execute("Write a market brief", task_category="research")

    receipt = swarm.trace[-1]

    # Stable identifier for the task, independent of its free-text prompt.
    assert "task_id" in receipt

    # Candidate agents + the exact bid inputs each agent submitted.
    assert {"researcher", "summarizer"} == {b["agent_id"] for b in receipt["bids"]}
    for bid in receipt["bids"]:
        assert "estimated_cost" in bid
        assert "estimated_quality" in bid

    # The reputation snapshot each bid was scored against, so a reviewer can
    # tell whether a win came from a real track record or a stale prior.
    for bid in receipt["bids"]:
        assert "reputation_prior" in bid
        assert "mean_quality" in bid["reputation_prior"]
        assert "version" in bid["reputation_prior"]

    # Selected vs. rejected roles must both be first-class, not derived.
    assert receipt["selected_roles"] == ["researcher"]
    assert receipt["rejected_roles"] == ["summarizer"]

    # Budget allocated vs. actually consumed by the winner(s).
    assert receipt["budget_allocated"] == 100
    assert "budget_consumed" in receipt

    # Where the outcome score came from: caller override, extracted from the
    # winning agent's own output, or a self-reported bid-value fallback. This
    # client returns neither actual_quality nor reward, so the source here
    # must be flagged as the self-reported fallback, not an external eval.
    assert receipt["outcome_score_source"] == "self_reported_bid_fallback"

    # The learning rule (and its version) that will use this outcome to
    # update future bids, so drift in the rule itself is auditable.
    assert receipt["learning_rule"]["name"] == "ema"
    assert receipt["learning_rule"]["learning_rate"] == 0.1
    assert "version" in receipt["learning_rule"]


def _swarm_with_two_agents(**market_kwargs):
    registry = InMemoryAgentRegistry()
    swarm = AgentSwarm(
        market=MarketPolicy(budget_per_task=100, seed=42, exploration_rate=0, **market_kwargs),
        registry=registry,
    )
    swarm.register_agent(
        AgentMetadata(id="researcher", model="mock", cost_multiplier=30.0, capacity=2),
        client=MarketClient("researcher", cost=30.0, quality=0.96),
    )
    swarm.register_agent(
        AgentMetadata(id="summarizer", model="mock", cost_multiplier=8.0, capacity=2),
        client=MarketClient("summarizer", cost=8.0, quality=0.74),
    )
    return swarm


async def test_self_reported_bid_cannot_ground_reputation_on_its_own():
    """A winning agent's own estimated_quality may appear in the receipt as
    evidence, but it must never advance reputation as grounded on its own."""
    swarm = _swarm_with_two_agents()
    await swarm.execute("Write a market brief", task_category="research")

    receipt = swarm.trace[-1]
    # The self-reported bid value is still recorded as evidence...
    assert receipt["actual_quality"] > 0
    assert receipt["outcome_score_source"] == "self_reported_bid_fallback"
    # ...but it is not grounded, and it did not mark the reputation grounded.
    assert receipt["outcome_signal_grounded"] is False
    snapshot = swarm.reputation.get("researcher", "research")
    assert snapshot.attempts == 1
    assert snapshot.grounded is False
    assert snapshot.grounded_fraction == 0.0


async def test_external_quality_override_grounds_the_outcome():
    swarm = _swarm_with_two_agents()
    await swarm.execute("Write a market brief", task_category="research", quality=0.9)

    receipt = swarm.trace[-1]
    assert receipt["outcome_score_source"] == "caller_override"
    assert receipt["outcome_signal_grounded"] is True
    snapshot = swarm.reputation.get("researcher", "research")
    assert snapshot.grounded is True
    assert snapshot.grounded_fraction == 1.0


async def test_require_grounded_reward_noops_ungrounded_update_but_default_does_not():
    # Default behaviour: an ungrounded outcome still updates reputation.
    lenient = _swarm_with_two_agents()
    await lenient.execute("brief", task_category="research")
    assert lenient.trace[-1]["reputation_updated"] is True
    assert lenient.reputation.get("researcher", "research").attempts == 1

    # Strict mode: the same ungrounded outcome no-ops the reputation update.
    strict = _swarm_with_two_agents(require_grounded_reward=True)
    await strict.execute("brief", task_category="research")
    assert strict.trace[-1]["reputation_updated"] is False
    assert strict.reputation.get("researcher", "research").attempts == 0

    # ...but a grounded (externally-supplied) outcome updates even in strict
    # mode. Use a fresh swarm so the per-agent task-share cap doesn't reshuffle
    # which agent wins the second auction.
    fresh = _swarm_with_two_agents(require_grounded_reward=True)
    await fresh.execute("brief", task_category="research", quality=0.88)
    assert fresh.trace[-1]["reputation_updated"] is True
    winner = fresh.trace[-1]["winners"][0]
    assert fresh.reputation.get(winner, "research").attempts == 1


def test_trace_is_indexable_and_backward_compatible_callable():
    swarm = _swarm_with_two_agents()
    # Empty trace is an indexable (falsy) sequence.
    assert list(swarm.trace) == []
    # The pre-2.0.1 method-style call still works and returns receipts.
    assert swarm.trace() == []


def test_signal_source_round_trips_from_receipt():
    # The receipt origin strings map back onto the two-tier SignalSource.
    assert GroundedSignal.external(0.9).source is SignalSource.EXTERNAL_OVERRIDE
    assert GroundedSignal.self_reported(0.9).source is SignalSource.SELF_REPORTED
