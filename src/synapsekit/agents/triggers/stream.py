"""Kafka and Redis Streams event-driven agent execution."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Literal

from .cron import TriggerResult

Backend = Literal["redis", "kafka"]


class StreamTrigger:
    """Consume messages from Redis Streams or Kafka and dispatch to an agent.

    Install:
    - Redis backend: ``pip install synapsekit[stream-trigger]``  (``redis>=5.0``)
    - Kafka backend: ``pip install synapsekit[stream-trigger]``  (``aiokafka>=0.10``)

    Parameters
    ----------
    agent_fn:
        Async callable ``(message: dict) -> str`` called for each message.
    backend:
        ``"redis"`` (default) or ``"kafka"``.
    topic:
        Redis stream key or Kafka topic name.
    group_name:
        Consumer group name (default ``"synapsekit"``).
    consumer_name:
        Consumer name / member id (default ``"agent-1"``).
    redis_url:
        Redis connection URL, e.g. ``"redis://localhost:6379"``.
    kafka_brokers:
        Comma-separated Kafka broker addresses, e.g. ``"localhost:9092"``.
    batch_size:
        Number of messages to fetch per poll cycle (default ``1``).
    poll_interval:
        Seconds to sleep between empty polls (default ``1.0``).
    result_sink:
        Optional async/sync callable receiving :class:`TriggerResult` objects.
    """

    def __init__(
        self,
        agent_fn: Callable[[dict[str, Any]], Awaitable[str]],
        *,
        backend: Backend = "redis",
        topic: str,
        group_name: str = "synapsekit",
        consumer_name: str = "agent-1",
        redis_url: str | None = None,
        kafka_brokers: str | None = None,
        batch_size: int = 1,
        poll_interval: float = 1.0,
        result_sink: Callable[[TriggerResult], Any] | None = None,
    ) -> None:
        if backend not in {"redis", "kafka"}:
            raise ValueError("backend must be 'redis' or 'kafka'")
        self._agent_fn = agent_fn
        self._backend: Backend = backend
        self._topic = topic
        self._group_name = group_name
        self._consumer_name = consumer_name
        self._redis_url = redis_url or "redis://localhost:6379"
        self._kafka_brokers = kafka_brokers or "localhost:9092"
        self._batch_size = batch_size
        self._poll_interval = poll_interval
        self._result_sink = result_sink
        self._stop_event: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        # Backend clients / consumers (set in start())
        self._redis_client: Any = None
        self._kafka_consumer: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin consuming messages in a background task."""
        if self._backend == "redis":
            await self._init_redis()
        else:
            await self._init_kafka()
        self._stop_event.clear()
        self._task = asyncio.create_task(self._consume_loop(), name="StreamTrigger")

    async def stop(self) -> None:
        """Stop consuming and clean up connections."""
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._redis_client is not None:
            await self._redis_client.aclose()
            self._redis_client = None
        if self._kafka_consumer is not None:
            await self._kafka_consumer.stop()
            self._kafka_consumer = None

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------

    async def _init_redis(self) -> None:
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "redis package required: pip install synapsekit[stream-trigger]"
            ) from None
        self._redis_client = aioredis.from_url(self._redis_url, decode_responses=True)
        # Create consumer group (ignore if already exists)
        with contextlib.suppress(Exception):
            await self._redis_client.xgroup_create(
                self._topic, self._group_name, id="$", mkstream=True
            )

    async def _consume_loop_redis(self) -> None:
        while not self._stop_event.is_set():
            results = await self._redis_client.xreadgroup(
                groupname=self._group_name,
                consumername=self._consumer_name,
                streams={self._topic: ">"},
                count=self._batch_size,
                block=int(self._poll_interval * 1000),
            )
            if not results:
                continue
            for _stream, messages in results:
                for message_id, fields in messages:
                    await self._dispatch(fields, ack_id=message_id)

    async def _ack_redis(self, message_id: str) -> None:
        await self._redis_client.xack(self._topic, self._group_name, message_id)

    # ------------------------------------------------------------------
    # Kafka
    # ------------------------------------------------------------------

    async def _init_kafka(self) -> None:
        try:
            from aiokafka import AIOKafkaConsumer  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "aiokafka package required: pip install synapsekit[stream-trigger]"
            ) from None
        self._kafka_consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._kafka_brokers,
            group_id=self._group_name,
            client_id=self._consumer_name,
            enable_auto_commit=False,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        )
        await self._kafka_consumer.start()

    async def _consume_loop_kafka(self) -> None:
        consumer = self._kafka_consumer
        while not self._stop_event.is_set():
            try:
                records = await asyncio.wait_for(
                    consumer.getmany(max_records=self._batch_size),
                    timeout=self._poll_interval,
                )
            except asyncio.TimeoutError:
                continue
            for _tp, messages in records.items():
                for msg in messages:
                    payload = msg.value if isinstance(msg.value, dict) else {}
                    await self._dispatch(payload, ack_fn=consumer.commit)

    # ------------------------------------------------------------------
    # Shared consume loop
    # ------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        if self._backend == "redis":
            await self._consume_loop_redis()
        else:
            await self._consume_loop_kafka()

    async def _dispatch(
        self,
        payload: dict[str, Any],
        ack_id: str | None = None,
        ack_fn: Any = None,
    ) -> None:
        scheduled_for = datetime.now(tz=timezone.utc)
        started_at = datetime.now(tz=timezone.utc)
        output: str | None = None
        error_text: str | None = None

        try:
            output = await self._agent_fn(payload)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"

        # Acknowledge message
        if ack_id is not None:
            await self._ack_redis(ack_id)
        elif ack_fn is not None:
            maybe = ack_fn()
            if inspect.isawaitable(maybe):
                await maybe

        finished_at = datetime.now(tz=timezone.utc)
        result = TriggerResult(
            scheduled_for=scheduled_for,
            started_at=started_at,
            finished_at=finished_at,
            input_text=json.dumps(payload),
            output=output,
            error=error_text,
            metadata={
                "backend": self._backend,
                "topic": self._topic,
                "group_name": self._group_name,
            },
        )

        if self._result_sink is not None:
            maybe_sink = self._result_sink(result)
            if inspect.isawaitable(maybe_sink):
                await maybe_sink
