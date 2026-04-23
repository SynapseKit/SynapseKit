from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """Async token-bucket rate limiter.

    Tokens are added at a fixed rate (``requests_per_minute / 60`` per second).
    Each call to :meth:`acquire` consumes one token, blocking if none are
    available.

    The wait happens **outside** the lock so that multiple concurrent callers
    can each compute their own sleep duration independently without serialising
    behind a single sleeper.
    """

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be >= 1")
        self._rpm = requests_per_minute
        self._tokens = float(requests_per_minute)
        self._max_tokens = float(requests_per_minute)
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it.

        The lock is released before sleeping so concurrent callers are not
        serialised behind a single waiter.
        """
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Compute how long to wait, then release the lock before sleeping
                wait = (1.0 - self._tokens) / self._rate

            # Sleep outside the lock — other callers can check/acquire freely
            await asyncio.sleep(wait)
