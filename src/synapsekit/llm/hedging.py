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

    async def _stream_hedged(
        self,
        stream_factory: Callable[[BaseLLM], AsyncGenerator[str]],
    ) -> AsyncGenerator[str]:
        self.stats.total_requests += 1
        primary_gen = stream_factory(self.primary)
        task_primary: asyncio.Task[str] = asyncio.create_task(primary_gen.__anext__())
        primary_failed = False
        primary_exc: Exception | None = None

        try:
            first_token = await asyncio.wait_for(
                asyncio.shield(task_primary), timeout=self.hedge_delay
            )
            self.stats.primary_won += 1
            self._last_winner = "primary"
            try:
                yield first_token
                async for token in primary_gen:
                    yield token
            finally:
                with contextlib.suppress(Exception):
                    await primary_gen.aclose()
            return
        except TimeoutError:
            pass
        except StopAsyncIteration:
            self.stats.primary_won += 1
            self._last_winner = "primary"
            with contextlib.suppress(Exception):
                await primary_gen.aclose()
            return
        except Exception as exc:
            primary_failed = True
            primary_exc = exc
            with contextlib.suppress(Exception):
                await primary_gen.aclose()

        self.stats.hedged_requests += 1
        secondary_gen = stream_factory(self.secondary)

        if primary_failed:
            self.stats.hedge_won += 1
            self._last_winner = "secondary"
            try:
                async for token in secondary_gen:
                    yield token
                return
            except Exception as sec_exc:
                raise sec_exc from primary_exc
            finally:
                with contextlib.suppress(Exception):
                    await secondary_gen.aclose()

        # Both streams are racing for the first token
        task_secondary: asyncio.Task[str] = asyncio.create_task(secondary_gen.__anext__())
        pending: set[asyncio.Task[str]] = {task_primary, task_secondary}
        winner_gen: AsyncGenerator[str] | None = None
        winner_token: str | None = None
        last_exc: Exception | None = None

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    tok = task.result()
                    if task is task_primary:
                        winner_gen = primary_gen
                        self.stats.primary_won += 1
                        self._last_winner = "primary"
                        for p in pending:
                            p.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await p
                        with contextlib.suppress(Exception):
                            await secondary_gen.aclose()
                    else:
                        winner_gen = secondary_gen
                        self.stats.hedge_won += 1
                        self._last_winner = "secondary"
                        for p in pending:
                            p.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await p
                        with contextlib.suppress(Exception):
                            await primary_gen.aclose()
                    winner_token = tok
                    break
                except StopAsyncIteration:
                    if task is task_primary:
                        winner_gen = primary_gen
                        self.stats.primary_won += 1
                        self._last_winner = "primary"
                        for p in pending:
                            p.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await p
                        with contextlib.suppress(Exception):
                            await secondary_gen.aclose()
                    else:
                        winner_gen = secondary_gen
                        self.stats.hedge_won += 1
                        self._last_winner = "secondary"
                        for p in pending:
                            p.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await p
                        with contextlib.suppress(Exception):
                            await primary_gen.aclose()
                    winner_token = ""
                    break
                except Exception as exc:
                    last_exc = exc
            if winner_gen is not None:
                break

        if winner_gen is None:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("All hedged streaming attempts failed")

        try:
            if winner_token:
                yield winner_token
            async for token in winner_gen:
                yield token
        finally:
            with contextlib.suppress(Exception):
                await winner_gen.aclose()

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
