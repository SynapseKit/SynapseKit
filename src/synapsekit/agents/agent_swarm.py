"""Market-based agent swarm orchestration."""

from __future__ import annotations

import asyncio
import inspect
import itertools
import math
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, cast

from ..provenance import GroundedSignal, SignalSource
from .agent_registry import (
    REPUTATION_LEARNING_RULE,
    REPUTATION_LEARNING_RULE_VERSION,
    AgentMetadata,
    AgentRegistry,
    AgentRegistryBackend,
    InMemoryAgentRegistry,
    Reputation,
    ReputationSnapshot,
)
from .federation import AgentFederation, LocalAgentClient


class _TraceList(list):
    """List of auction receipts.

    ``AgentSwarm.trace`` is exposed as an indexable sequence (``swarm.trace[-1]``
    is the latest receipt). This subclass stays callable — ``swarm.trace()`` —
    for backward compatibility with the pre-2.0.1 method-style API, returning a
    shallow copy of the receipts.
    """

    def __call__(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self]


class AuctionType(str, Enum):
    """Auction modes supported by :class:`AgentSwarm`."""

    SEALED_BID = "sealed_bid"
    ENGLISH = "english"
    VICKREY = "vickrey"
    MULTI_WINNER = "multi_winner"
    COALITION = "coalition"


@dataclass(slots=True)
class Bid:
    """An agent's estimate for executing a task."""

    agent_id: str
    estimated_cost: float
    estimated_quality: float
    confidence: float = 1.0
    task_category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.agent_id = str(self.agent_id)
        self.task_category = str(self.task_category or "general")
        self.estimated_cost = max(0.0, float(self.estimated_cost))
        self.estimated_quality = min(1.0, max(0.0, float(self.estimated_quality)))
        self.confidence = min(1.0, max(0.0, float(self.confidence)))
        self.metadata = dict(self.metadata)
        if not self.agent_id:
            raise ValueError("Bid agent_id must be a non-empty string.")


@dataclass(slots=True)
class AuctionResult:
    """Decision record for one market auction."""

    task: str
    task_category: str
    auction_type: AuctionType
    bids: list[Bid]
    winners: list[Bid]
    scores: dict[str, float]
    settlement_cost: float
    latency_ms: float
    coalition: bool = False

    @property
    def winner_ids(self) -> list[str]:
        return [bid.agent_id for bid in self.winners]


@dataclass(slots=True)
class SwarmResult:
    """Execution result returned by :class:`AgentSwarm`."""

    output: Any
    winners: list[str]
    auction: AuctionResult
    results: dict[str, Any]
    reward: float
    actual_cost: float
    actual_quality: float

    def __str__(self) -> str:
        return str(self.output)


class AgentBidder(Protocol):
    """Protocol for agents that submit their own market bids."""

    def bid(self, task: str, **kwargs: Any) -> Bid | dict[str, Any]:
        """Return a bid for ``task``."""


@dataclass(frozen=True, slots=True)
class BidStrategy:
    """Scoring strategy used by market auctions.

    The class exposes factory methods so users can write
    ``BidStrategy.cost_quality_pareto()`` as in the issue sketch.
    """

    name: str
    exploration_weight: float = 0.15
    samples: int = 8

    @classmethod
    def cost_quality_pareto(cls) -> BidStrategy:
        return cls("cost_quality_pareto", exploration_weight=0.05)

    @classmethod
    def thompson_sampling(cls, *, samples: int = 8) -> BidStrategy:
        return cls("thompson_sampling", exploration_weight=0.10, samples=max(1, int(samples)))

    @classmethod
    def ucb(cls, *, exploration_weight: float = 0.35) -> BidStrategy:
        return cls("ucb", exploration_weight=max(0.0, float(exploration_weight)))

    @classmethod
    def from_value(cls, value: BidStrategy | str | None) -> BidStrategy:
        if value is None:
            return cls.cost_quality_pareto()
        if isinstance(value, BidStrategy):
            return value
        name = str(value).replace("-", "_").lower()
        if name in {"cost_quality", "cost_quality_pareto", "pareto"}:
            return cls.cost_quality_pareto()
        if name in {"thompson", "thompson_sampling"}:
            return cls.thompson_sampling()
        if name == "ucb":
            return cls.ucb()
        raise ValueError(f"Unknown bid strategy: {value}")

    def score(
        self,
        bid: Bid,
        reputation: ReputationSnapshot,
        *,
        budget: float,
        total_executions: int,
        rng: random.Random,
    ) -> float:
        cost_penalty = 0.0 if budget <= 0 else bid.estimated_cost / budget
        quality = 0.75 * bid.estimated_quality + 0.25 * reputation.mean_quality
        base = quality * max(0.05, bid.confidence) - (0.4 * cost_penalty)

        if self.name == "ucb":
            visits = max(1, reputation.attempts)
            exploration = math.sqrt(math.log(max(2, total_executions + 1)) / visits)
            return base + self.exploration_weight * exploration

        if self.name == "thompson_sampling":
            sampled = (
                sum(
                    rng.betavariate(reputation.quality_alpha, reputation.quality_beta)
                    for _ in range(self.samples)
                )
                / self.samples
            )
            return 0.55 * base + 0.45 * sampled

        return base


@dataclass(slots=True)
class MarketPolicy:
    """Market configuration for :class:`AgentSwarm`."""

    bid_strategy: BidStrategy | str | None = None
    auction_type: AuctionType | str = AuctionType.SEALED_BID
    budget_per_task: float = 10_000.0
    learning_rate: float = 0.1
    seed: int | None = None
    max_winners: int = 1
    coalition_size: int = 2
    exploration_rate: float = 0.05
    cold_start_quality: float = 0.6
    max_agent_task_share: float = 0.70
    require_grounded_reward: bool = False

    def __post_init__(self) -> None:
        self.bid_strategy = BidStrategy.from_value(self.bid_strategy)
        self.auction_type = self._coerce_auction_type(self.auction_type)
        self.budget_per_task = max(0.0, float(self.budget_per_task))
        self.learning_rate = min(1.0, max(0.0, float(self.learning_rate)))
        self.max_winners = max(1, int(self.max_winners))
        self.coalition_size = max(2, int(self.coalition_size))
        self.exploration_rate = min(1.0, max(0.0, float(self.exploration_rate)))
        self.cold_start_quality = min(1.0, max(0.0, float(self.cold_start_quality)))
        self.max_agent_task_share = min(1.0, max(0.0, float(self.max_agent_task_share)))
        self.require_grounded_reward = bool(self.require_grounded_reward)

    @staticmethod
    def _coerce_auction_type(value: AuctionType | str) -> AuctionType:
        if isinstance(value, AuctionType):
            return value
        normalised = str(value).replace("-", "_").lower()
        aliases = {
            "sealed": "sealed_bid",
            "sealedbid": "sealed_bid",
            "multi": "multi_winner",
            "multiwinner": "multi_winner",
        }
        try:
            return AuctionType(aliases.get(normalised, normalised))
        except ValueError as exc:
            raise ValueError(f"Unknown auction type: {value}") from exc


class CoalitionFormer:
    """Build deterministic coalitions from ranked bids."""

    def __init__(self, *, max_size: int = 2) -> None:
        self.max_size = max(2, int(max_size))

    def form(self, bids: list[Bid], scores: dict[str, float]) -> list[Bid]:
        if len(bids) < self.max_size:
            raise LookupError(
                f"Coalition requires at least {self.max_size} agents; got {len(bids)}."
            )
        if len(bids) == self.max_size:
            return list(bids)

        best: tuple[float, tuple[Bid, ...]] | None = None
        for combo in itertools.combinations(bids, self.max_size):
            score = sum(scores[bid.agent_id] for bid in combo) / len(combo)
            score += min(bid.estimated_quality for bid in combo) * 0.05
            if best is None or score > best[0]:
                best = (score, combo)
        return list(best[1]) if best is not None else []


class AgentSwarm:
    """Market-based orchestration for a registry of agents."""

    def __init__(
        self,
        agents: Iterable[Any] | None = None,
        *,
        market: MarketPolicy | None = None,
        registry: AgentRegistry | AgentRegistryBackend | InMemoryAgentRegistry | None = None,
        clients: dict[str, Any] | None = None,
        reputation: Reputation | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.market = market or MarketPolicy()
        self.registry = registry or InMemoryAgentRegistry()
        self.reputation = reputation or Reputation()
        self.metrics = metrics
        self.federation = AgentFederation(self.registry, clients=clients)
        self._rng = random.Random(self.market.seed)
        self._trace: list[dict[str, Any]] = []
        self._wins_by_agent: dict[str, int] = {}
        self._executions = 0

        for index, agent in enumerate(agents or []):
            metadata, client = self._coerce_agent(agent, index)
            self.federation.register_agent(metadata, client=client)

    def register_agent(self, agent: AgentMetadata, *, client: Any | None = None) -> AgentMetadata:
        return self.federation.register_agent(agent, client=client)

    register = register_agent

    def add_client(self, agent_id: str, client: Any) -> None:
        self.federation.add_client(agent_id, client)

    async def run(self, task: str, **kwargs: Any) -> SwarmResult:
        return await self.execute(task, **kwargs)

    async def execute(
        self,
        task: str,
        *,
        task_category: str | None = None,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
        reward: float | None = None,
        quality: float | None = None,
        **kwargs: Any,
    ) -> SwarmResult:
        category = self._task_category(task, task_category)
        candidates = self.federation.discover(
            tools=tools,
            tags=tags,
            min_capacity=min_capacity,
            healthy_only=healthy_only,
        )
        if not candidates:
            raise LookupError("No agents match the swarm discovery filters.")

        auction = await self.auction(task, candidates, task_category=category)
        if not auction.winners:
            raise LookupError("No agents submitted an eligible market bid.")

        # Capture the reputation snapshot each bid was scored against *before*
        # any outcome is recorded, so the receipt shows the prior the market
        # actually used — not the post-update state.
        priors = {bid.agent_id: self.reputation.get(bid.agent_id, category) for bid in auction.bids}

        task_id = f"task-{self._executions + 1}"
        results = await self._execute_winners(task, auction.winners, kwargs)
        output = self._combine_results(results, auction)
        actual_cost = self._actual_cost(output, auction)
        quality_signal = self._quality_signal(output, auction, quality)
        actual_quality = quality_signal.value
        actual_reward = self._actual_reward(output, actual_cost, actual_quality, reward)

        # Skeptical-by-default: an ungrounded (self-reported) outcome still
        # updates reputation as it always has. Only the opt-in strict mode
        # no-ops the reputation update when the signal isn't externally grounded.
        reputation_updated = quality_signal.grounded or not self.market.require_grounded_reward

        for winner in auction.winners:
            if reputation_updated:
                self.reputation.record_outcome(
                    winner.agent_id,
                    category,
                    cost=actual_cost / len(auction.winners),
                    quality=actual_quality,
                    reward=actual_reward,
                    won=True,
                    learning_rate=self.market.learning_rate,
                    quality_signal=quality_signal,
                )
            self._wins_by_agent[winner.agent_id] = self._wins_by_agent.get(winner.agent_id, 0) + 1
            await self._settle_bidder(winner, auction, actual_cost, actual_quality, actual_reward)
            self._record_win_metric(winner, category, auction, actual_cost, actual_reward)

        self._executions += 1
        winner_ids = set(auction.winner_ids)
        self._trace.append(
            {
                "task": task,
                "task_id": task_id,
                "task_category": category,
                "auction_type": auction.auction_type.value,
                "bids": [
                    {
                        "agent_id": bid.agent_id,
                        "estimated_cost": bid.estimated_cost,
                        "estimated_quality": bid.estimated_quality,
                        "confidence": bid.confidence,
                        "score": auction.scores.get(bid.agent_id, 0.0),
                        "reputation_prior": self._prior_view(priors.get(bid.agent_id)),
                    }
                    for bid in auction.bids
                ],
                "winners": auction.winner_ids,
                "selected_roles": auction.winner_ids,
                "rejected_roles": [
                    bid.agent_id for bid in auction.bids if bid.agent_id not in winner_ids
                ],
                "settlement_cost": auction.settlement_cost,
                "budget_allocated": self.market.budget_per_task,
                "budget_consumed": actual_cost,
                "reward": actual_reward,
                "actual_cost": actual_cost,
                "actual_quality": actual_quality,
                "outcome_score_source": quality_signal.provenance.get("origin", "unknown"),
                "outcome_signal_grounded": quality_signal.grounded,
                "reputation_updated": reputation_updated,
                "learning_rule": {
                    "name": REPUTATION_LEARNING_RULE,
                    "learning_rate": self.market.learning_rate,
                    "version": REPUTATION_LEARNING_RULE_VERSION,
                },
            }
        )

        return SwarmResult(
            output=output,
            winners=auction.winner_ids,
            auction=auction,
            results=results,
            reward=actual_reward,
            actual_cost=actual_cost,
            actual_quality=actual_quality,
        )

    async def auction(
        self,
        task: str,
        candidates: list[AgentMetadata],
        *,
        task_category: str | None = None,
    ) -> AuctionResult:
        category = self._task_category(task, task_category)
        started = time.perf_counter()
        bids = await self._collect_bids(task, candidates, category)
        eligible = self._eligible_bids(bids)
        scores = self._score_bids(eligible, category)
        winners = self._select_winners(eligible, scores)
        latency_ms = (time.perf_counter() - started) * 1000.0
        settlement = self._settlement_cost(winners, eligible, scores)

        result = AuctionResult(
            task=task,
            task_category=category,
            auction_type=cast(AuctionType, self.market.auction_type),
            bids=eligible,
            winners=winners,
            scores=scores,
            settlement_cost=settlement,
            latency_ms=latency_ms,
            coalition=self.market.auction_type == AuctionType.COALITION,
        )
        self._record_auction_metric(category, result)
        return result

    @property
    def trace(self) -> _TraceList:
        """Auction receipts, newest last. ``swarm.trace[-1]`` is the latest.

        Each receipt is a replayable role-allocation record: ``task_id``, every
        ``bid`` with the ``reputation_prior`` it was scored against,
        ``selected_roles`` / ``rejected_roles``, ``budget_allocated`` /
        ``budget_consumed``, the ``outcome_score_source`` (and whether it was
        grounded), and the ``learning_rule`` that will fold the outcome into
        future bids. The returned list is also callable — ``swarm.trace()`` —
        for backward compatibility with the pre-2.0.1 method API.
        """
        return _TraceList(dict(entry) for entry in self._trace)

    @staticmethod
    def _prior_view(snapshot: ReputationSnapshot | None) -> dict[str, Any]:
        """A compact, replayable view of the reputation prior a bid was scored
        against — enough to tell a real track record from a stale one."""
        if snapshot is None:
            return {"mean_quality": 0.0, "attempts": 0, "version": 0, "grounded_fraction": 0.0}
        return {
            "mean_quality": snapshot.mean_quality,
            "mean_reward": snapshot.mean_reward,
            "attempts": snapshot.attempts,
            "wins": snapshot.wins,
            "version": snapshot.version,
            "grounded_fraction": snapshot.grounded_fraction,
        }

    def trace_to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        if not self._trace:
            lines.append('    empty["No swarm executions recorded"]')
            return "\n".join(lines)

        for index, entry in enumerate(self._trace, start=1):
            task_node = f"task{index}"
            lines.append(f'    {task_node}["Task {index}: {entry["task_category"]}"]')
            for bid in entry["bids"]:
                agent_node = self._mermaid_id(f"agent_{index}_{bid['agent_id']}")
                label = f"{bid['agent_id']}\\nscore={bid['score']:.3f}"
                lines.append(f'    {agent_node}["{label}"]')
                lines.append(f"    {task_node} --> {agent_node}")
            for winner in entry["winners"]:
                winner_node = self._mermaid_id(f"agent_{index}_{winner}")
                result_node = self._mermaid_id(f"result_{index}_{winner}")
                lines.append(
                    f'    {result_node}["winner: {winner}\\nreward={entry["reward"]:.3f}"]'
                )
                lines.append(f"    {winner_node} ==> {result_node}")
        return "\n".join(lines)

    def _coerce_agent(self, agent: Any, index: int) -> tuple[AgentMetadata, Any | None]:
        if isinstance(agent, AgentMetadata):
            return agent, None

        config = getattr(agent, "config", None)
        agent_id = (
            getattr(agent, "agent_id", None)
            or getattr(config, "agent_id", None)
            or getattr(agent, "name", None)
            or f"agent-{index + 1}"
        )
        model = getattr(getattr(config, "llm", None), "model", None) or getattr(
            getattr(getattr(config, "llm", None), "config", None), "model", None
        )
        metadata = AgentMetadata(
            id=str(agent_id),
            model=str(model or type(agent).__name__),
            tools=[getattr(tool, "name", str(tool)) for tool in getattr(config, "tools", [])],
            tags=list(getattr(agent, "tags", [])),
        )
        return metadata, agent

    async def _collect_bids(
        self,
        task: str,
        candidates: list[AgentMetadata],
        category: str,
    ) -> list[Bid]:
        bids: list[Bid] = []
        for agent in candidates:
            bid = await self._bid_for_agent(task, agent, category)
            if bid is not None:
                bids.append(bid)
                self._record_bid_metric(bid, category)
        return bids

    async def _bid_for_agent(
        self,
        task: str,
        agent: AgentMetadata,
        category: str,
    ) -> Bid | None:
        bidder = self._bidder_for(agent.id)
        if bidder is None:
            return self._synthetic_bid(agent, category)

        reputation = self.reputation.get(agent.id, category)
        try:
            result = bidder.bid(
                task,
                agent=agent,
                reputation=reputation,
                market=self.market,
                task_category=category,
            )
        except TypeError:
            result = bidder.bid(task)
        if inspect.isawaitable(result):
            result = await result
        return self._coerce_bid(result, agent, category)

    def _bidder_for(self, agent_id: str) -> Any | None:
        client = self.federation.clients.get(agent_id)
        if client is None:
            return None
        if hasattr(client, "bid"):
            return client
        if isinstance(client, LocalAgentClient) and hasattr(client.executor, "bid"):
            return client.executor
        return None

    def _coerce_bid(self, result: Any, agent: AgentMetadata, category: str) -> Bid | None:
        if result is None:
            return None
        if isinstance(result, Bid):
            if result.agent_id == agent.id and result.task_category == category:
                return result
            return Bid(
                agent_id=agent.id,
                estimated_cost=result.estimated_cost,
                estimated_quality=result.estimated_quality,
                confidence=result.confidence,
                task_category=category,
                metadata=result.metadata,
            )
        if isinstance(result, dict):
            estimated_cost = cast(Any, result.get("estimated_cost", result.get("cost", 0.0)))
            estimated_quality = cast(
                Any,
                result.get(
                    "estimated_quality",
                    result.get("quality", self.market.cold_start_quality),
                ),
            )
            return Bid(
                agent_id=agent.id,
                estimated_cost=float(estimated_cost),
                estimated_quality=float(estimated_quality),
                confidence=float(result.get("confidence", 1.0)),
                task_category=category,
                metadata=dict(result.get("metadata", {})),
            )
        raise TypeError("Agent bid must be a Bid, dict, or None.")

    def _synthetic_bid(self, agent: AgentMetadata, category: str) -> Bid:
        snapshot = self.reputation.get(agent.id, category)
        if snapshot.attempts:
            quality = snapshot.mean_quality
            confidence = min(1.0, 0.45 + snapshot.attempts / 20.0)
            cost = snapshot.avg_cost if snapshot.avg_cost is not None else agent.cost_multiplier
        else:
            quality = min(1.0, self.market.cold_start_quality + min(agent.capacity, 10) * 0.015)
            confidence = 0.35
            cost = agent.cost_multiplier

        return Bid(
            agent_id=agent.id,
            estimated_cost=max(0.0, cost),
            estimated_quality=quality,
            confidence=confidence,
            task_category=category,
            metadata={"source": "metadata"},
        )

    def _eligible_bids(self, bids: list[Bid]) -> list[Bid]:
        if not bids:
            return []
        in_budget = [bid for bid in bids if bid.estimated_cost <= self.market.budget_per_task]
        eligible = in_budget or bids
        if len(eligible) < 2 or self._executions == 0:
            return eligible

        capped = [
            bid
            for bid in eligible
            if self._agent_share(bid.agent_id) < self.market.max_agent_task_share
        ]
        return capped or eligible

    def _score_bids(self, bids: list[Bid], category: str) -> dict[str, float]:
        strategy = cast(BidStrategy, self.market.bid_strategy)
        scores: dict[str, float] = {}
        for bid in bids:
            snapshot = self.reputation.get(bid.agent_id, category)
            scores[bid.agent_id] = strategy.score(
                bid,
                snapshot,
                budget=max(1.0, self.market.budget_per_task),
                total_executions=self._executions,
                rng=self._rng,
            )

        if (
            len(bids) > 1
            and self.market.exploration_rate > 0
            and self._rng.random() < self.market.exploration_rate
        ):
            chosen = self._rng.choice(bids)
            scores[chosen.agent_id] = max(scores.values()) + 1.0
        return scores

    def _select_winners(self, bids: list[Bid], scores: dict[str, float]) -> list[Bid]:
        if not bids:
            return []
        ordered = sorted(
            bids,
            key=lambda bid: (scores[bid.agent_id], -bid.estimated_cost, bid.agent_id),
            reverse=True,
        )

        if self.market.auction_type == AuctionType.MULTI_WINNER:
            return ordered[: min(self.market.max_winners, len(ordered))]
        if self.market.auction_type == AuctionType.COALITION:
            return CoalitionFormer(max_size=self.market.coalition_size).form(ordered, scores)
        if self.market.auction_type == AuctionType.ENGLISH:
            affordable = [
                bid for bid in ordered if bid.estimated_cost <= self.market.budget_per_task
            ]
            return [affordable[0] if affordable else ordered[0]]
        return [ordered[0]]

    def _settlement_cost(
        self, winners: list[Bid], bids: list[Bid], scores: dict[str, float]
    ) -> float:
        if not winners:
            return 0.0
        if self.market.auction_type == AuctionType.VICKREY and len(bids) > 1:
            ordered_by_price = sorted(bids, key=lambda bid: bid.estimated_cost, reverse=True)
            second = ordered_by_price[1]
            return min(self.market.budget_per_task, second.estimated_cost)
        return min(self.market.budget_per_task, sum(bid.estimated_cost for bid in winners))

    async def _execute_winners(
        self,
        task: str,
        winners: list[Bid],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        async def invoke(bid: Bid) -> tuple[str, Any]:
            client = self.federation.get_client(bid.agent_id)
            result = client.run(task, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return bid.agent_id, result

        pairs = await asyncio.gather(*(invoke(bid) for bid in winners))
        return dict(pairs)

    def _combine_results(self, results: dict[str, Any], auction: AuctionResult) -> Any:
        if len(results) == 1:
            return next(iter(results.values()))
        return {
            "mode": "coalition" if auction.coalition else "multi_winner",
            "winners": auction.winner_ids,
            "results": results,
        }

    def _actual_cost(self, output: Any, auction: AuctionResult) -> float:
        value = self._extract_number(output, "actual_cost", "cost", "cost_tokens")
        if value is not None:
            return max(0.0, value)
        return auction.settlement_cost

    def _actual_quality(self, output: Any, auction: AuctionResult, override: float | None) -> float:
        return self._quality_signal(output, auction, override).value

    def _quality_signal(
        self, output: Any, auction: AuctionResult, override: float | None
    ) -> GroundedSignal:
        """Resolve the outcome quality *and where it came from*.

        A caller-supplied ``override`` is the only externally-grounded source. A
        ``quality``/``score`` field read out of the winning agent's own output,
        and the bid-estimate fallback used when nothing else is available, are
        both the agent describing itself — so both are ``SELF_REPORTED`` (see
        :mod:`synapsekit.provenance` for why there is no middle tier).
        """
        if override is not None:
            return GroundedSignal(
                value=min(1.0, max(0.0, float(override))),
                source=SignalSource.EXTERNAL_OVERRIDE,
                provenance={"origin": "caller_override"},
            )
        value = self._extract_number(output, "actual_quality", "quality", "score")
        if value is not None:
            return GroundedSignal(
                value=min(1.0, max(0.0, value)),
                source=SignalSource.SELF_REPORTED,
                provenance={"origin": "output_field"},
            )
        if not auction.winners:
            return GroundedSignal(
                value=0.0,
                source=SignalSource.SELF_REPORTED,
                provenance={"origin": "no_winner"},
            )
        fallback = sum(bid.estimated_quality for bid in auction.winners) / len(auction.winners)
        return GroundedSignal(
            value=fallback,
            source=SignalSource.SELF_REPORTED,
            provenance={"origin": "self_reported_bid_fallback"},
        )

    def _actual_reward(
        self,
        output: Any,
        actual_cost: float,
        actual_quality: float,
        override: float | None,
    ) -> float:
        if override is not None:
            return float(override)
        value = self._extract_number(output, "reward")
        if value is not None:
            return value
        cost_penalty = actual_cost / max(1.0, self.market.budget_per_task)
        return actual_quality - cost_penalty

    @staticmethod
    def _extract_number(output: Any, *keys: str) -> float | None:
        if isinstance(output, dict):
            for key in keys:
                value = output.get(key)
                if isinstance(value, int | float):
                    return float(value)
            nested = output.get("results")
            if isinstance(nested, dict):
                values = [
                    AgentSwarm._extract_number(value, *keys)
                    for value in nested.values()
                    if isinstance(value, dict)
                ]
                clean = [value for value in values if value is not None]
                if clean:
                    return sum(clean) / len(clean)
        return None

    async def _settle_bidder(
        self,
        bid: Bid,
        auction: AuctionResult,
        actual_cost: float,
        actual_quality: float,
        reward: float,
    ) -> None:
        bidder = self._bidder_for(bid.agent_id)
        settle = getattr(bidder, "settle", None)
        if settle is None:
            return
        result = settle(
            bid=bid,
            auction=auction,
            actual_cost=actual_cost,
            actual_quality=actual_quality,
            reward=reward,
        )
        if inspect.isawaitable(result):
            await result

    def _record_bid_metric(self, bid: Bid, category: str) -> None:
        recorder = getattr(self.metrics, "record_swarm_bid", None)
        if recorder is not None:
            auction_type = cast(AuctionType, self.market.auction_type)
            recorder(agent_id=bid.agent_id, task_category=category, auction_type=auction_type.value)

    def _record_auction_metric(self, category: str, auction: AuctionResult) -> None:
        recorder = getattr(self.metrics, "record_swarm_auction", None)
        if recorder is not None:
            recorder(
                auction_type=auction.auction_type.value,
                task_category=category,
                bid_count=len(auction.bids),
                latency_ms=auction.latency_ms,
            )

    def _record_win_metric(
        self,
        bid: Bid,
        category: str,
        auction: AuctionResult,
        actual_cost: float,
        reward: float,
    ) -> None:
        recorder = getattr(self.metrics, "record_swarm_win", None)
        if recorder is not None:
            recorder(
                agent_id=bid.agent_id,
                task_category=category,
                auction_type=auction.auction_type.value,
                reward=reward,
                settlement_cost=actual_cost,
            )

    def _agent_share(self, agent_id: str) -> float:
        if self._executions <= 0:
            return 0.0
        return self._wins_by_agent.get(agent_id, 0) / self._executions

    @staticmethod
    def _task_category(task: str, explicit: str | None) -> str:
        if explicit:
            return str(explicit)
        words = [word.strip(".,:;!?()[]{}").lower() for word in task.split()]
        return next((word for word in words if word), "general")

    @staticmethod
    def _mermaid_id(value: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in value)
