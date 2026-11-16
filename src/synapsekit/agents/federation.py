"""Federated agent routing and client adapters."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Iterable
from enum import Enum
from typing import Any, Protocol, cast

from .agent_registry import (
    AgentMetadata,
    AgentRegistry,
    AgentRegistryBackend,
    InMemoryAgentRegistry,
)
from .negotiation import Message, Negotiation, NegotiationStatus


class AgentClient(Protocol):
    """Interface used by AgentFederation to invoke an agent."""

    def run(self, prompt: str, **kwargs: Any) -> Any:
        """Run the agent for a prompt."""


class LocalAgentClient:
    """AgentClient adapter for local AgentExecutor-like objects or callables."""

    def __init__(self, executor: Any) -> None:
        self.executor = executor

    async def run(self, prompt: str, **kwargs: Any) -> Any:
        target = self.executor
        if callable(target) and not any(
            hasattr(target, name)
            for name in ("arun", "run", "aexecute", "execute", "ainvoke", "invoke")
        ):
            result = target(prompt, **kwargs)
        else:
            for method_name in ("arun", "run", "aexecute", "execute", "ainvoke", "invoke"):
                method = getattr(target, method_name, None)
                if method is not None:
                    result = method(prompt, **kwargs)
                    break
            else:
                raise TypeError(
                    "LocalAgentClient requires an executor with run/arun, "
                    "execute/aexecute, invoke/ainvoke, or a callable executor."
                )

        if inspect.isawaitable(result):
            return await result
        return result


class RoutingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    CAPACITY_AWARE = "capacity_aware"
    COST_AWARE = "cost_aware"
    MARKET = "market"


AgentRoutingStrategy = RoutingStrategy


class AgentFederation:
    """Discover agents and route prompts across a registry."""

    def __init__(
        self,
        registry: AgentRegistry | AgentRegistryBackend | InMemoryAgentRegistry | None = None,
        *,
        clients: dict[str, Any] | None = None,
        default_strategy: RoutingStrategy | str = RoutingStrategy.ROUND_ROBIN,
    ) -> None:
        self.registry = registry or InMemoryAgentRegistry()
        self.clients: dict[str, AgentClient] = {
            str(agent_id): self._coerce_client(client)
            for agent_id, client in (clients or {}).items()
        }
        self.default_strategy = self._normalise_strategy(default_strategy)
        self._round_robin_offsets: dict[tuple[str, tuple[str, ...]], int] = {}

    def register_agent(
        self,
        agent: AgentMetadata,
        *,
        client: Any | None = None,
    ) -> AgentMetadata:
        registered = self.registry.register(agent)
        if client is not None:
            self.clients[registered.id] = self._coerce_client(client)
        return registered

    register = register_agent

    def unregister_agent(self, agent_id: str) -> bool:
        self.clients.pop(str(agent_id), None)
        return self.registry.unregister(agent_id)

    unregister = unregister_agent

    def add_client(self, agent_id: str, client: Any) -> None:
        self.clients[str(agent_id)] = self._coerce_client(client)

    def discover(
        self,
        *,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
    ) -> list[AgentMetadata]:
        return self.registry.discover(
            tools=tools,
            tags=tags,
            min_capacity=min_capacity,
            healthy_only=healthy_only,
        )

    discover_agents = discover

    def select_agent(
        self,
        *,
        strategy: RoutingStrategy | str | None = None,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
    ) -> AgentMetadata:
        candidates = self.discover(
            tools=tools,
            tags=tags,
            min_capacity=min_capacity,
            healthy_only=healthy_only,
        )
        if not candidates:
            raise LookupError("No agents match the discovery filters.")

        selected_strategy = self._normalise_strategy(strategy or self.default_strategy)
        if selected_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin(candidates, "round_robin")
        if selected_strategy == RoutingStrategy.CAPACITY_AWARE:
            max_capacity = max(agent.capacity for agent in candidates)
            best = [agent for agent in candidates if agent.capacity == max_capacity]
            return self._round_robin(best, "capacity_aware")
        if selected_strategy == RoutingStrategy.COST_AWARE:
            min_cost = min(agent.cost_multiplier for agent in candidates)
            best = [agent for agent in candidates if agent.cost_multiplier == min_cost]
            return self._round_robin(best, "cost_aware")
        if selected_strategy == RoutingStrategy.MARKET:
            raise ValueError("Market routing requires AgentFederation.run(..., strategy='market').")

        raise ValueError(f"Unsupported routing strategy: {selected_strategy}")

    route = select_agent
    select = select_agent

    def _ranked_candidates(
        self,
        *,
        strategy: RoutingStrategy,
        tools: Iterable[str] | str | None,
        tags: Iterable[str] | str | None,
        min_capacity: int | None,
        healthy_only: bool,
    ) -> list[AgentMetadata]:
        """Return all matching agents ordered best-first for *strategy*.

        Used by ``run`` to fail over through the ranked list when a chosen
        agent crashes or times out.
        """
        candidates = self.discover(
            tools=tools,
            tags=tags,
            min_capacity=min_capacity,
            healthy_only=healthy_only,
        )
        if not candidates:
            raise LookupError("No agents match the discovery filters.")

        if strategy == RoutingStrategy.CAPACITY_AWARE:
            return sorted(candidates, key=lambda a: a.capacity, reverse=True)
        if strategy == RoutingStrategy.COST_AWARE:
            return sorted(candidates, key=lambda a: a.cost_multiplier)
        # ROUND_ROBIN: rotate by the current offset so the primary pick matches
        # select_agent, then fall through the remaining agents in order.
        ids = tuple(agent.id for agent in candidates)
        key = ("round_robin", ids)
        offset = self._round_robin_offsets.get(key, 0)
        self._round_robin_offsets[key] = offset + 1
        n = len(candidates)
        return [candidates[(offset + i) % n] for i in range(n)]

    def _record_failure(self, agent: AgentMetadata) -> None:
        """Best-effort mark of an agent as failed via the registry/reputation API."""
        for method_name in ("mark_unhealthy", "report_failure", "record_failure"):
            method = getattr(self.registry, method_name, None)
            if callable(method):
                with contextlib.suppress(Exception):
                    method(agent.id)
                return

    async def negotiate(
        self, initial: Message, *, max_rounds: int = 10, deliver_terminal: bool = True
    ) -> Negotiation:
        """Exchange proposals between two registered, explicitly capable clients.

        ``negotiate(message)`` returns a response. By default,
        both clients must also provide ``receive_negotiation(message)``; it is
        called on the terminal recipient and may be async. Its exceptions propagate.
        ``deliver_terminal=False`` records the result without notifying that party.
        No routing, execution, or payment is performed.
        """
        if not isinstance(deliver_terminal, bool):
            raise TypeError("deliver_terminal must be a bool.")
        session = Negotiation(initial, max_rounds=max_rounds)
        parties = {}
        for agent_id in (initial.sender, initial.recipient):
            agent = self.registry.get(agent_id)
            if agent is None:
                raise KeyError(f"Unknown agent id: {agent_id}")
            if not self.registry.is_healthy(agent_id):
                raise LookupError(f"Agent is not healthy: {agent_id}")
            parties[agent_id] = self.get_client(agent)

        handlers = {}
        receivers = {}
        for agent_id, client in parties.items():
            target = client.executor if isinstance(client, LocalAgentClient) else client
            handler = getattr(target, "negotiate", None)
            if not callable(handler):
                raise TypeError(f"Agent {agent_id} does not support negotiate(message).")
            handlers[agent_id] = handler
            if deliver_terminal:
                receiver = getattr(target, "receive_negotiation", None)
                if not callable(receiver):
                    raise TypeError(
                        f"Agent {agent_id} does not support receive_negotiation(message)."
                    )
                receivers[agent_id] = receiver

        while session.status == NegotiationStatus.OPEN:
            proposal = session.messages[-1]
            response = handlers[proposal.recipient](proposal)
            if inspect.isawaitable(response):
                response = await response
            session.apply(response)
        if deliver_terminal:
            delivered = receivers[response.recipient](response)
            if inspect.isawaitable(delivered):
                await delivered
        return session

    async def run(
        self,
        prompt: str,
        *,
        agent_id: str | None = None,
        strategy: RoutingStrategy | str | None = None,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        selected_strategy = self._normalise_strategy(strategy or self.default_strategy)
        if (agent_id is not None or selected_strategy != RoutingStrategy.MARKET) and any(
            key in kwargs
            for key in ("negotiated_agreements", "negotiation_buyer", "negotiation_requirements")
        ):
            raise ValueError("Negotiation routing inputs require market strategy without agent_id.")
        if agent_id is None and selected_strategy == RoutingStrategy.MARKET:
            from .agent_swarm import AgentSwarm, MarketPolicy

            market = kwargs.pop("market", None)
            if market is not None and not isinstance(market, MarketPolicy):
                raise TypeError("market must be a MarketPolicy when strategy='market'.")
            swarm = AgentSwarm(
                market=market,
                registry=self.registry,
                clients=dict(self.clients),
                reputation=kwargs.pop("reputation", None),
                metrics=kwargs.pop("metrics", None),
            )
            return await swarm.execute(
                prompt,
                tools=tools,
                tags=tags,
                min_capacity=min_capacity,
                healthy_only=healthy_only,
                **kwargs,
            )

        if agent_id is not None:
            agent_record = self.registry.get(agent_id)
            if agent_record is None:
                raise KeyError(f"Unknown agent id: {agent_id}")
            if healthy_only and not self.registry.is_healthy(agent_record.id):
                raise LookupError(f"Agent is not healthy: {agent_record.id}")
            return await self._invoke_one(agent_record, prompt, timeout, **kwargs)

        # No explicit agent: build the ranked candidate list and fail over
        # through it. A crash or timeout on one agent re-routes to the next
        # healthy candidate instead of failing the whole task.
        candidates = self._ranked_candidates(
            strategy=selected_strategy,
            tools=tools,
            tags=tags,
            min_capacity=min_capacity,
            healthy_only=healthy_only,
        )
        last_exc: BaseException | None = None
        for agent in candidates:
            try:
                return await self._invoke_one(agent, prompt, timeout, **kwargs)
            except Exception as exc:  # includes asyncio.TimeoutError
                last_exc = exc
                self._record_failure(agent)
                continue
        raise RuntimeError(
            f"All {len(candidates)} candidate agents failed for the task."
        ) from last_exc

    async def _invoke_one(
        self,
        agent: AgentMetadata,
        prompt: str,
        timeout: float | None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a single agent's client, awaiting with an optional timeout."""
        client = self.get_client(agent)
        result = client.run(prompt, **kwargs)
        if inspect.isawaitable(result):
            if timeout is not None:
                return await asyncio.wait_for(result, timeout=timeout)
            return await result
        return result

    async def execute(self, prompt: str, **kwargs: Any) -> Any:
        return await self.run(prompt, **kwargs)

    def get_client(self, agent: AgentMetadata | str) -> AgentClient:
        agent_id = agent if isinstance(agent, str) else agent.id
        client = self.clients.get(str(agent_id))
        if client is None and not isinstance(agent, str) and agent.endpoint is not None:
            client = self.clients.get(agent.endpoint)
        if client is None:
            raise KeyError(f"No AgentClient registered for agent: {agent_id}")
        return client

    def _round_robin(self, candidates: list[AgentMetadata], strategy_key: str) -> AgentMetadata:
        ids = tuple(agent.id for agent in candidates)
        key = (strategy_key, ids)
        offset = self._round_robin_offsets.get(key, 0)
        self._round_robin_offsets[key] = offset + 1
        return candidates[offset % len(candidates)]

    def _normalise_strategy(self, strategy: RoutingStrategy | str) -> RoutingStrategy:
        if isinstance(strategy, RoutingStrategy):
            return strategy
        value = str(strategy).replace("-", "_").lower()
        aliases = {
            "roundrobin": "round_robin",
            "rr": "round_robin",
            "capacity": "capacity_aware",
            "capacity_aware_routing": "capacity_aware",
            "cost": "cost_aware",
            "cost_aware_routing": "cost_aware",
            "market_based": "market",
            "market_routing": "market",
            "swarm": "market",
        }
        value = aliases.get(value, value)
        try:
            return RoutingStrategy(value)
        except ValueError as exc:
            raise ValueError(f"Unknown routing strategy: {strategy}") from exc

    def _coerce_client(self, client: Any) -> AgentClient:
        if hasattr(client, "run"):
            return cast(AgentClient, client)
        return LocalAgentClient(client)
