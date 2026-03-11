from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

# Error substrings that indicate auth/permission issues — never retry these.
_NO_RETRY_PATTERNS = (
    "authentication",
    "api_key",
    "api key",
    "unauthorized",
    "forbidden",
    "permission",
    "invalid_api_key",
)


async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args: object,
    max_retries: int = 0,
    delay: float = 1.0,
    **kwargs: object,
) -> T:
    """
    Call *fn* with exponential backoff.

    Skips retrying on auth/permission errors (pattern-matched).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            err_lower = str(exc).lower()
            if any(p in err_lower for p in _NO_RETRY_PATTERNS):
                raise
            if attempt < max_retries:
                await asyncio.sleep(delay * (2**attempt))
    raise last_exc  # type: ignore[misc]
