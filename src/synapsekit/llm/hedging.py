"""HedgedLLM — Request Hedging for p99 Tail Latency Mitigation in LLM Inference.

Request Hedging (based on Google's 'The Tail at Scale' architecture) solves
the p99 tail-latency problem in production LLM applications.

When making an LLM request:
1. Immediately dispatches the request to the primary LLM.
2. If the primary does not respond (or yield its first streaming chunk) within
   a configured SLA deadline (``hedge_delay``, e.g. 1.5 seconds), a second
   "hedged" request is dispatched concurrently to the secondary LLM.
3. Whichever request finishes first (or yields the first token first) wins.
4. The slower in-flight request is cancelled immediately to free resources.
5. If the primary fails outright before the hedge delay expires, the secondary
   is triggered immediately without waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .base import BaseLLM, LLMConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def _drain_stream(gen: AsyncGenerator[str], queue: asyncio.Queue[tuple[str, Any]]) -> None:
    """Fully drive a candidate stream inside a single task, funnelling tokens to a queue.

    The generator's whole lifetime (first ``__anext__`` through close) runs in this
    task's context, so contextvar-based instrumentation (e.g. ``synapsekit.observe``
    spans) sets and resets its token in the *same* context. Driving ``__anext__()``
    directly inside ``asyncio.create_task`` would instead copy the context, breaking
    the span reset. The queue carries ``("item", token)`` per chunk, ``("done", None)``
    on normal exhaustion, and ``("error", exc)`` on failure.
    """
    try:
        async for tok in gen:
            await queue.put(("item", tok))
        await queue.put(("done", None))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Surfaced to the consumer via the queue rather than raised in-task.
        await queue.put(("error", exc))
    finally:
        with contextlib.suppress(Exception):
            await gen.aclose()


@dataclass
class HedgingStats:
    """Observability and FinOps metrics for request hedging."""

    total_requests: int = 0
    hedged_requests: int = 0
    primary_won: int = 0
    hedge_won: int = 0

    @property
    def hedge_rate(self) -> float:
        """Fraction of total requests that triggered a hedged backup call."""
        if self.total_requests == 0:
            return 0.0
        return self.hedged_requests / self.total_requests

    @property
    def hedge_win_rate(self) -> float:
        """Fraction of hedged calls where the hedge model won over the lagging primary."""
        if self.hedged_requests == 0:
            return 0.0
        return self.hedge_won / self.hedged_requests


@dataclass
class HedgingConfig:
    """Configuration for HedgedLLM."""

    primary: BaseLLM
    secondary: BaseLLM
    hedge_delay: float = 1.5  # seconds

    def __post_init__(self) -> None:
        if self.hedge_delay <= 0:
            raise ValueError(f"hedge_delay must be positive, got {self.hedge_delay!r}")


class HedgedLLM(BaseLLM):
    """Wraps two BaseLLM models to hedge against p99 tail latency.

    Example::

        from synapsekit.llm import HedgedLLM, HedgingConfig, OpenAILLM, AnthropicLLM

        primary = OpenAILLM(LLMConfig(model="gpt-4o", api_key="sk-..."))
        secondary = AnthropicLLM(LLMConfig(model="claude-3-5-sonnet", api_key="sk-..."))

        hedged = HedgedLLM(HedgingConfig(
            primary=primary,
            secondary=secondary,
            hedge_delay=1.5,
        ))

        # If primary takes > 1.5s, secondary is dispatched concurrently.
        # Whichever responds first wins!
        answer = await hedged.generate("Complex prompt...")
        print(hedged.stats.hedge_rate)
    """

    def __init__(self, config: HedgingConfig) -> None:
        super().__init__(
            LLMConfig(
                model=f"hedged({config.primary.config.model},{config.secondary.config.model})",
                api_key="",
                provider="hedged",
            )
        )
        self.hedging_config = config
        self.primary = config.primary
        self.secondary = config.secondary
        self.hedge_delay = config.hedge_delay
        self.stats = HedgingStats()
        self._last_winner: str | None = None
        self.supports_multimodal: bool = bool(
            getattr(config.primary, "supports_multimodal", False)
            and getattr(config.secondary, "supports_multimodal", False)
        )

    @property
    def last_winner(self) -> str | None:
        """Return the winner of the last request ('primary' or 'secondary')."""
        return self._last_winner

    async def _execute_hedged(self, run_fn: Callable[[BaseLLM], Any]) -> Any:
        self.stats.total_requests += 1

        primary_task = asyncio.create_task(run_fn(self.primary))
        primary_failed = False
        primary_exc: Exception | None = None

        try:
            res = await asyncio.wait_for(asyncio.shield(primary_task), timeout=self.hedge_delay)
            self.stats.primary_won += 1
            self._last_winner = "primary"
            return res
        except TimeoutError:
            # Primary exceeded hedge_delay SLA
            pass
        except Exception as exc:
            primary_failed = True
            primary_exc = exc

        # Dispatch hedged backup request
        self.stats.hedged_requests += 1
        secondary_task = asyncio.create_task(run_fn(self.secondary))

        if primary_failed:
            try:
                res = await secondary_task
                self.stats.hedge_won += 1
                self._last_winner = "secondary"
                return res
            except Exception as sec_exc:
                raise sec_exc from primary_exc

        pending = {primary_task, secondary_task}
        last_exc: Exception | None = None

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    res = task.result()
                    for p in pending:
                        p.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await p
                    if task is primary_task:
                        self.stats.primary_won += 1
                        self._last_winner = "primary"
                    else:
                        self.stats.hedge_won += 1
                        self._last_winner = "secondary"
                    return res
                except Exception as exc:
                    last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All hedged LLM tasks failed")

    async def generate(self, prompt: str, **kw: Any) -> str:
        """Generate response with tail-latency hedging."""
        return str(await self._execute_hedged(lambda llm: llm.generate(prompt, **kw)))

    async def generate_with_messages(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        """Generate response from messages list with tail-latency hedging."""
        return str(
            await self._execute_hedged(lambda llm: llm.generate_with_messages(messages, **kw))
        )

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute tool calling with tail-latency hedging."""
        return dict(await self._execute_hedged(lambda llm: llm.call_with_tools(messages, tools)))

    @staticmethod
    async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
        """Cancel a task (if still running) and await it, swallowing its teardown."""
        if task is None:
            return
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _emit_queue(
        self,
        first_msg: tuple[str, Any],
        queue: asyncio.Queue[tuple[str, Any]],
        drain_task: asyncio.Task[None],
    ) -> AsyncGenerator[str]:
        """Yield the winner's tokens from its queue, then tear its drain task down."""
        try:
            msg = first_msg
            while True:
                kind, value = msg
                if kind == "item":
                    yield value
                elif kind == "done":
                    return
                else:  # "error"
                    raise value
                msg = await queue.get()
        finally:
            await self._cancel_task(drain_task)

    async def _stream_hedged(
        self,
        stream_factory: Callable[[BaseLLM], AsyncGenerator[str]],
    ) -> AsyncGenerator[str]:
        self.stats.total_requests += 1

        primary_gen = stream_factory(self.primary)
        primary_q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        primary_task = asyncio.create_task(_drain_stream(primary_gen, primary_q))

        # Race the primary's first signal against the hedge-delay SLA deadline.
        # The getter is left pending (not cancelled) on timeout so no first token
        # is lost: it simply joins the hedge race below.
        get_primary: asyncio.Task[tuple[str, Any]] = asyncio.ensure_future(primary_q.get())
        timer = asyncio.ensure_future(asyncio.sleep(self.hedge_delay))
        watch: set[asyncio.Future[Any]] = {get_primary, timer}
        done, _ = await asyncio.wait(watch, return_when=asyncio.FIRST_COMPLETED)

        primary_exc: Exception | None = None
        primary_dead = False

        if get_primary in done:
            await self._cancel_task(timer)
            msg = get_primary.result()
            if msg[0] != "error":
                # Primary responded within the SLA: it wins outright, no hedge.
                self.stats.primary_won += 1
                self._last_winner = "primary"
                async for item in self._emit_queue(msg, primary_q, primary_task):
                    yield item
                return
            # Primary failed before the deadline: hedge immediately.
            primary_exc = msg[1]
            primary_dead = True
            await self._cancel_task(primary_task)
        # else: deadline hit; `get_primary` stays pending for the race.

        # Dispatch the hedged backup request.
        self.stats.hedged_requests += 1
        secondary_gen = stream_factory(self.secondary)
        secondary_q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        secondary_task = asyncio.create_task(_drain_stream(secondary_gen, secondary_q))

        if primary_dead:
            # Primary already dead; cancel its stray getter and lean on the secondary.
            await self._cancel_task(get_primary)
            sec_first = await secondary_q.get()
            if sec_first[0] == "error":
                await self._cancel_task(secondary_task)
                raise sec_first[1] from primary_exc
            self.stats.hedge_won += 1
            self._last_winner = "secondary"
            async for item in self._emit_queue(sec_first, secondary_q, secondary_task):
                yield item
            return

        # True race: primary still in flight (timed out) vs freshly dispatched secondary.
        queues = {"primary": primary_q, "secondary": secondary_q}
        drain_tasks = {"primary": primary_task, "secondary": secondary_task}
        getters: dict[str, asyncio.Task[tuple[str, Any]]] = {
            "primary": get_primary,
            "secondary": asyncio.ensure_future(secondary_q.get()),
        }
        winner: str | None = None
        winner_msg: tuple[str, Any] | None = None
        last_exc: Exception | None = None

        while getters and winner is None:
            race_done, _ = await asyncio.wait(getters.values(), return_when=asyncio.FIRST_COMPLETED)
            for name in list(getters):
                if getters[name] not in race_done:
                    continue
                result = getters[name].result()
                if result[0] == "error":
                    last_exc = result[1]
                    del getters[name]
                    await self._cancel_task(drain_tasks.pop(name))
                else:
                    winner = name
                    winner_msg = result
                    break

        if winner is None or winner_msg is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("All hedged streaming attempts failed")

        # Cancel the losing racer (if any) and its drain task.
        for name, getter in getters.items():
            if name != winner:
                await self._cancel_task(getter)
                await self._cancel_task(drain_tasks[name])

        if winner == "primary":
            self.stats.primary_won += 1
        else:
            self.stats.hedge_won += 1
        self._last_winner = winner

        async for item in self._emit_queue(winner_msg, queues[winner], drain_tasks[winner]):
            yield item

    async def stream(self, prompt: str, **kw: Any) -> AsyncGenerator[str]:
        """Stream tokens with tail-latency hedging on first token arrival."""
        async for token in self._stream_hedged(lambda llm: llm.stream(prompt, **kw)):
            yield token

    async def stream_with_messages(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncGenerator[str]:
        """Stream tokens from messages with tail-latency hedging on first token arrival."""
        async for token in self._stream_hedged(
            lambda llm: llm.stream_with_messages(messages, **kw)
        ):
            yield token

    async def aclose(self) -> None:
        """Close both underlying LLM clients."""
        await asyncio.gather(
            self.primary.aclose(),
            self.secondary.aclose(),
            return_exceptions=True,
        )
