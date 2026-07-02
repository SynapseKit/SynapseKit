"""Agent metadata registry backends."""

from __future__ import annotations

import json
import threading
import time
from builtins import list as builtin_list
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


def _normalise_strings(values: Iterable[str] | str | None) -> builtin_list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


@dataclass(slots=True)
class AgentMetadata:
    """Metadata used to discover and route work to an agent."""

    id: str
    model: str
    tools: builtin_list[str] = field(default_factory=list)
    capacity: int = 1
    cost_multiplier: float = 1.0
    tags: builtin_list[str] = field(default_factory=list)
    endpoint: str | None = None
    last_heartbeat: float | None = None

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.model = str(self.model)
        if not self.id:
            raise ValueError("Agent id must be a non-empty string.")
        if not self.model:
            raise ValueError("Agent model must be a non-empty string.")

        self.tools = _normalise_strings(self.tools)
        self.tags = _normalise_strings(self.tags)
        self.capacity = int(self.capacity)
        self.cost_multiplier = float(self.cost_multiplier)

        if self.capacity < 0:
            raise ValueError("Agent capacity must be greater than or equal to zero.")
        if self.cost_multiplier < 0:
            raise ValueError("Agent cost_multiplier must be greater than or equal to zero.")
        if self.endpoint is not None:
            self.endpoint = str(self.endpoint)
        if self.last_heartbeat is not None:
            self.last_heartbeat = float(self.last_heartbeat)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "tools": list(self.tools),
            "capacity": self.capacity,
            "cost_multiplier": self.cost_multiplier,
            "tags": list(self.tags),
            "endpoint": self.endpoint,
            "last_heartbeat": self.last_heartbeat,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMetadata:
        return cls(
            id=data["id"],
            model=data["model"],
            tools=data.get("tools", []),
            capacity=data.get("capacity", 1),
            cost_multiplier=data.get("cost_multiplier", 1.0),
            tags=data.get("tags", []),
            endpoint=data.get("endpoint"),
            last_heartbeat=data.get("last_heartbeat"),
        )

    def copy(self) -> AgentMetadata:
        return AgentMetadata.from_dict(self.to_dict())

    def is_healthy(self, stale_timeout: float, now: float | None = None) -> bool:
        if self.last_heartbeat is None:
            return False
        current_time = time.time() if now is None else now
        return current_time - self.last_heartbeat <= stale_timeout


@dataclass(slots=True)
class ReputationSnapshot:
    """Per-agent, per-task-category reputation used by market routing."""

    agent_id: str
    task_category: str = "general"
    attempts: int = 0
    wins: int = 0
    avg_cost: float | None = None
    mean_quality: float = 0.5
    mean_reward: float = 0.0
    quality_alpha: float = 1.0
    quality_beta: float = 1.0

    def __post_init__(self) -> None:
        self.agent_id = str(self.agent_id)
        self.task_category = str(self.task_category or "general")
        self.attempts = max(0, int(self.attempts))
        self.wins = max(0, int(self.wins))
        self.avg_cost = None if self.avg_cost is None else max(0.0, float(self.avg_cost))
        self.mean_quality = min(1.0, max(0.0, float(self.mean_quality)))
        self.mean_reward = float(self.mean_reward)
        self.quality_alpha = max(0.001, float(self.quality_alpha))
        self.quality_beta = max(0.001, float(self.quality_beta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "task_category": self.task_category,
            "attempts": self.attempts,
            "wins": self.wins,
            "avg_cost": self.avg_cost,
            "mean_quality": self.mean_quality,
            "mean_reward": self.mean_reward,
            "quality_alpha": self.quality_alpha,
            "quality_beta": self.quality_beta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReputationSnapshot:
        return cls(
            agent_id=data["agent_id"],
            task_category=data.get("task_category", "general"),
            attempts=data.get("attempts", 0),
            wins=data.get("wins", 0),
            avg_cost=data.get("avg_cost", None),
            mean_quality=data.get("mean_quality", 0.5),
            mean_reward=data.get("mean_reward", 0.0),
            quality_alpha=data.get("quality_alpha", 1.0),
            quality_beta=data.get("quality_beta", 1.0),
        )

    def copy(self) -> ReputationSnapshot:
        return ReputationSnapshot.from_dict(self.to_dict())


class Reputation:
    """In-memory reputation store for market-based agent routing."""

    def __init__(self, *, default_quality: float = 0.5) -> None:
        self.default_quality = min(1.0, max(0.0, float(default_quality)))
        self._snapshots: dict[tuple[str, str], ReputationSnapshot] = {}
        self._lock = threading.RLock()

    def get(self, agent_id: str, task_category: str = "general") -> ReputationSnapshot:
        key = self._key(agent_id, task_category)
        with self._lock:
            snapshot = self._snapshots.get(key)
            if snapshot is None:
                return ReputationSnapshot(
                    agent_id=str(agent_id),
                    task_category=str(task_category or "general"),
                    mean_quality=self.default_quality,
                )
            return snapshot.copy()

    def set(self, snapshot: ReputationSnapshot) -> ReputationSnapshot:
        clean = snapshot.copy()
        with self._lock:
            self._snapshots[self._key(clean.agent_id, clean.task_category)] = clean
        return clean.copy()

    def record_outcome(
        self,
        agent_id: str,
        task_category: str = "general",
        *,
        cost: float,
        quality: float,
        reward: float,
        won: bool = True,
        learning_rate: float = 0.1,
    ) -> ReputationSnapshot:
        learning_rate = min(1.0, max(0.0, float(learning_rate)))
        snapshot = self.get(agent_id, task_category)
        previous_attempts = snapshot.attempts
        snapshot.attempts += 1
        if won:
            snapshot.wins += 1

        quality = min(1.0, max(0.0, float(quality)))
        cost = max(0.0, float(cost))
        reward = float(reward)
        if previous_attempts == 0:
            snapshot.avg_cost = cost
            snapshot.mean_quality = quality
            snapshot.mean_reward = reward
        else:
            snapshot.avg_cost = self._ema(snapshot.avg_cost, cost, learning_rate)
            snapshot.mean_quality = self._ema(snapshot.mean_quality, quality, learning_rate)
            snapshot.mean_reward = self._ema(snapshot.mean_reward, reward, learning_rate)

        snapshot.quality_alpha += quality
        snapshot.quality_beta += 1.0 - quality
        return self.set(snapshot)

    def list(self) -> builtin_list[ReputationSnapshot]:
        with self._lock:
            return [snapshot.copy() for snapshot in self._snapshots.values()]

    @staticmethod
    def _key(agent_id: str, task_category: str) -> tuple[str, str]:
        return (str(agent_id), str(task_category or "general"))

    @staticmethod
    def _ema(old: float, new: float, learning_rate: float) -> float:
        return (old * (1.0 - learning_rate)) + (new * learning_rate)


class RedisReputation(Reputation):
    """Redis-backed reputation store.

    Requires the optional redis extra, matching :class:`RedisAgentRegistry`.
    """

    def __init__(
        self,
        *,
        redis_client: Any | None = None,
        url: str | None = None,
        redis_url: str | None = None,
        prefix: str = "synapsekit:agent_reputation",
        default_quality: float = 0.5,
        **redis_kwargs: Any,
    ) -> None:
        super().__init__(default_quality=default_quality)
        self.prefix = prefix.rstrip(":")
        if redis_client is not None:
            self._redis = redis_client
            return

        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisReputation requires the optional redis dependency. "
                "Install it with `pip install synapsekit[redis]`."
            ) from exc

        redis_kwargs.setdefault("decode_responses", True)
        connection_url = url or redis_url
        if connection_url is not None:
            self._redis = redis.from_url(connection_url, **redis_kwargs)
        else:
            self._redis = redis.Redis(**redis_kwargs)

    def get(self, agent_id: str, task_category: str = "general") -> ReputationSnapshot:
        raw = self._redis.get(self._redis_key(agent_id, task_category))
        if raw is None:
            return ReputationSnapshot(
                agent_id=str(agent_id),
                task_category=str(task_category or "general"),
                mean_quality=self.default_quality,
            )
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return ReputationSnapshot.from_dict(json.loads(raw))

    def set(self, snapshot: ReputationSnapshot) -> ReputationSnapshot:
        clean = snapshot.copy()
        self._redis.set(
            self._redis_key(clean.agent_id, clean.task_category), json.dumps(clean.to_dict())
        )
        return clean.copy()

    def list(self) -> builtin_list[ReputationSnapshot]:
        snapshots: builtin_list[ReputationSnapshot] = []
        for key in self._redis.scan_iter(match=f"{self.prefix}:*"):
            raw = self._redis.get(key)
            if raw is None:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            snapshots.append(ReputationSnapshot.from_dict(json.loads(raw)))
        return sorted(snapshots, key=lambda item: (item.agent_id, item.task_category))

    def _redis_key(self, agent_id: str, task_category: str) -> str:
        return f"{self.prefix}:{agent_id}:{task_category or 'general'}"


def _matches_discovery_filters(
    agent: AgentMetadata,
    *,
    tools: Iterable[str] | str | None = None,
    tags: Iterable[str] | str | None = None,
    min_capacity: int | None = None,
    healthy_only: bool = True,
    stale_timeout: float = 30.0,
    now: float | None = None,
) -> bool:
    required_tools = set(_normalise_strings(tools))
    required_tags = set(_normalise_strings(tags))

    if required_tools and not required_tools.issubset(set(agent.tools)):
        return False
    if required_tags and not required_tags.issubset(set(agent.tags)):
        return False
    if min_capacity is not None and agent.capacity < min_capacity:
        return False
    if healthy_only:
        return agent.is_healthy(stale_timeout, now=now)
    return True


class AgentRegistryBackend(Protocol):
    def register(self, agent: AgentMetadata) -> AgentMetadata:
        pass

    def unregister(self, agent_id: str) -> bool:
        pass

    def get(self, agent_id: str) -> AgentMetadata | None:
        pass

    def list(self) -> builtin_list[AgentMetadata]:
        pass

    def heartbeat(self, agent_id: str, timestamp: float | None = None) -> AgentMetadata:
        pass

    def is_healthy(self, agent_id: str) -> bool:
        pass

    def discover(
        self,
        *,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
    ) -> builtin_list[AgentMetadata]:
        pass

    stale_timeout: float

    def prune_stale(self, *, stale_timeout: float | None = None) -> builtin_list[str]:
        pass


class InMemoryAgentRegistry:
    """In-memory agent registry with heartbeat-based health checks."""

    def __init__(
        self,
        *,
        stale_timeout: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.stale_timeout = float(stale_timeout)
        self._clock = clock or time.time
        self._agents: dict[str, AgentMetadata] = {}
        self._lock = threading.RLock()

    def register(self, agent: AgentMetadata) -> AgentMetadata:
        registered = agent.copy()
        if registered.last_heartbeat is None:
            registered.last_heartbeat = self._clock()
        with self._lock:
            self._agents[registered.id] = registered
        return registered.copy()

    register_agent = register

    def unregister(self, agent_id: str) -> bool:
        with self._lock:
            return self._agents.pop(str(agent_id), None) is not None

    unregister_agent = unregister

    def get(self, agent_id: str) -> AgentMetadata | None:
        with self._lock:
            agent = self._agents.get(str(agent_id))
            return None if agent is None else agent.copy()

    get_agent = get

    def list(self) -> builtin_list[AgentMetadata]:
        with self._lock:
            return [agent.copy() for agent in self._agents.values()]

    list_agents = list

    def heartbeat(
        self,
        agent_id: str,
        timestamp: float | None = None,
    ) -> AgentMetadata:
        with self._lock:
            agent = self._agents.get(str(agent_id))
            if agent is None:
                raise KeyError(f"Unknown agent id: {agent_id}")
            agent.last_heartbeat = self._clock() if timestamp is None else float(timestamp)
            return agent.copy()

    update_heartbeat = heartbeat

    def is_healthy(self, agent_id: str) -> bool:
        agent = self.get(agent_id)
        if agent is None:
            return False
        return agent.is_healthy(self.stale_timeout, now=self._clock())

    def discover(
        self,
        *,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
    ) -> builtin_list[AgentMetadata]:
        now = self._clock()
        return [
            agent
            for agent in self.list()
            if _matches_discovery_filters(
                agent,
                tools=tools,
                tags=tags,
                min_capacity=min_capacity,
                healthy_only=healthy_only,
                stale_timeout=self.stale_timeout,
                now=now,
            )
        ]

    discover_agents = discover

    def prune_stale(self, *, stale_timeout: float | None = None) -> builtin_list[str]:
        now = self._clock()
        timeout = self.stale_timeout if stale_timeout is None else float(stale_timeout)
        removed: builtin_list[str] = []
        with self._lock:
            for agent_id, agent in list(self._agents.items()):
                if not agent.is_healthy(timeout, now=now):
                    removed.append(agent_id)
                    del self._agents[agent_id]
        return removed

    prune_stale_agents = prune_stale


class RedisAgentRegistry:
    """Redis-backed agent registry.

    Requires the optional redis extra:
    ``pip install synapsekit[redis]``.
    """

    def __init__(
        self,
        *,
        redis_client: Any | None = None,
        url: str | None = None,
        redis_url: str | None = None,
        prefix: str = "synapsekit:agent_registry",
        stale_timeout: float = 30.0,
        **redis_kwargs: Any,
    ) -> None:
        self.prefix = prefix.rstrip(":")
        self.stale_timeout = float(stale_timeout)
        if redis_client is not None:
            self._redis = redis_client
            return

        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisAgentRegistry requires the optional redis dependency. "
                "Install it with `pip install synapsekit[redis]`."
            ) from exc

        redis_kwargs.setdefault("decode_responses", True)
        connection_url = url or redis_url
        if connection_url is not None:
            self._redis = redis.from_url(connection_url, **redis_kwargs)
        else:
            self._redis = redis.Redis(**redis_kwargs)

    def _key(self, agent_id: str) -> str:
        return f"{self.prefix}:{agent_id}"

    def _loads(self, raw: Any) -> AgentMetadata | None:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return AgentMetadata.from_dict(json.loads(raw))

    def _dumps(self, agent: AgentMetadata) -> str:
        return json.dumps(agent.to_dict(), sort_keys=True)

    def register(self, agent: AgentMetadata) -> AgentMetadata:
        registered = agent.copy()
        if registered.last_heartbeat is None:
            registered.last_heartbeat = time.time()
        self._redis.set(self._key(registered.id), self._dumps(registered))
        return registered.copy()

    register_agent = register

    def unregister(self, agent_id: str) -> bool:
        return bool(self._redis.delete(self._key(str(agent_id))))

    unregister_agent = unregister

    def get(self, agent_id: str) -> AgentMetadata | None:
        agent = self._loads(self._redis.get(self._key(str(agent_id))))
        return None if agent is None else agent.copy()

    get_agent = get

    def list(self) -> builtin_list[AgentMetadata]:
        agents: builtin_list[AgentMetadata] = []
        for key in self._redis.scan_iter(match=f"{self.prefix}:*"):
            agent = self._loads(self._redis.get(key))
            if agent is not None:
                agents.append(agent)
        return sorted(agents, key=lambda agent: agent.id)

    list_agents = list

    def heartbeat(
        self,
        agent_id: str,
        timestamp: float | None = None,
    ) -> AgentMetadata:
        agent = self.get(agent_id)
        if agent is None:
            raise KeyError(f"Unknown agent id: {agent_id}")
        agent.last_heartbeat = time.time() if timestamp is None else float(timestamp)
        self._redis.set(self._key(agent.id), self._dumps(agent))
        return agent.copy()

    update_heartbeat = heartbeat

    def is_healthy(self, agent_id: str) -> bool:
        agent = self.get(agent_id)
        if agent is None:
            return False
        return agent.is_healthy(self.stale_timeout)

    def discover(
        self,
        *,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
    ) -> builtin_list[AgentMetadata]:
        now = time.time()
        return [
            agent
            for agent in self.list()
            if _matches_discovery_filters(
                agent,
                tools=tools,
                tags=tags,
                min_capacity=min_capacity,
                healthy_only=healthy_only,
                stale_timeout=self.stale_timeout,
                now=now,
            )
        ]

    discover_agents = discover

    def prune_stale(self, *, stale_timeout: float | None = None) -> builtin_list[str]:
        now = time.time()
        timeout = self.stale_timeout if stale_timeout is None else float(stale_timeout)
        removed: builtin_list[str] = []
        for agent in self.list():
            if not agent.is_healthy(timeout, now=now):
                removed.append(agent.id)
                self.unregister(agent.id)
        return removed

    prune_stale_agents = prune_stale


class AgentRegistry:
    """Facade for selecting an agent registry backend.

    ``AgentRegistry()`` creates an in-memory registry. Use
    ``AgentRegistry(backend="redis", url="redis://...")`` for Redis.
    """

    def __init__(self, backend: str | Any = "memory", **kwargs: Any) -> None:
        self._backend: AgentRegistryBackend
        if isinstance(backend, str):
            backend_name = backend.replace("-", "_").lower()
            if backend_name in {"memory", "in_memory", "inmemory"}:
                self._backend = InMemoryAgentRegistry(**kwargs)
            elif backend_name == "redis":
                self._backend = RedisAgentRegistry(**kwargs)
            else:
                raise ValueError(f"Unknown agent registry backend: {backend}")
        else:
            self._backend = backend

    @classmethod
    def in_memory(cls, **kwargs: Any) -> AgentRegistry:
        return cls("memory", **kwargs)

    @classmethod
    def redis(cls, **kwargs: Any) -> AgentRegistry:
        return cls("redis", **kwargs)

    @property
    def backend(self) -> AgentRegistryBackend:
        return self._backend

    @property
    def stale_timeout(self) -> float:
        return float(self._backend.stale_timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def register(self, agent: AgentMetadata) -> AgentMetadata:
        return self._backend.register(agent)

    register_agent = register

    def unregister(self, agent_id: str) -> bool:
        return self._backend.unregister(agent_id)

    unregister_agent = unregister

    def get(self, agent_id: str) -> AgentMetadata | None:
        return self._backend.get(agent_id)

    get_agent = get

    def list(self) -> builtin_list[AgentMetadata]:
        return self._backend.list()

    list_agents = list

    def heartbeat(
        self,
        agent_id: str,
        timestamp: float | None = None,
    ) -> AgentMetadata:
        return self._backend.heartbeat(agent_id, timestamp=timestamp)

    update_heartbeat = heartbeat

    def is_healthy(self, agent_id: str) -> bool:
        return self._backend.is_healthy(agent_id)

    def discover(
        self,
        *,
        tools: Iterable[str] | str | None = None,
        tags: Iterable[str] | str | None = None,
        min_capacity: int | None = None,
        healthy_only: bool = True,
    ) -> builtin_list[AgentMetadata]:
        return self._backend.discover(
            tools=tools,
            tags=tags,
            min_capacity=min_capacity,
            healthy_only=healthy_only,
        )

    discover_agents = discover

    def prune_stale(self, *, stale_timeout: float | None = None) -> builtin_list[str]:
        return self._backend.prune_stale(stale_timeout=stale_timeout)

    prune_stale_agents = prune_stale
