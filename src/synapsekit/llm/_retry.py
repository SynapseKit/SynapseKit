from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

# HTTP status codes that are safe to retry.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value (seconds form only) into a float."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except (ValueError, AttributeError):
        return None
    return seconds if seconds >= 0 else None


def _is_retryable(exc: BaseException) -> bool:
    """Classify an exception as retryable based on type / HTTP status.

    Retries only on:
      * timeouts (``asyncio.TimeoutError``, ``TimeoutError``, ``httpx.TimeoutException``)
      * connection errors (``ConnectionError``, ``httpx.ConnectError``/``TransportError``)
      * HTTP responses with status 429 or 5xx

    Never retries other 4xx responses (auth, bad request, not found, etc.).
    """
    # Timeouts.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    # OpenAI wraps httpx connection and timeout failures in SDK-specific errors
    # that do not inherit from ConnectionError. Classify by the optional SDK's
    # exception type without importing the optional dependency here.
    if any(
        (cls.__module__ == "openai" or cls.__module__.startswith("openai."))
        and cls.__name__ in {"APIConnectionError", "APITimeoutError"}
        for cls in type(exc).__mro__
    ):
        return True

    # httpx errors — imported lazily so httpx stays an optional dependency.
    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore[assignment]

    if httpx is not None:
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in _RETRYABLE_STATUS
        # Broader transport-level failures (read/write/pool errors) are retryable,
        # but keep protocol/decoding errors (which won't fix themselves) out.
        if isinstance(exc, httpx.TransportError):
            return True

    # Objects that expose an HTTP ``status_code`` (many provider SDK errors do).
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS

    return False


def _retry_after_from_exc(exc: BaseException) -> float | None:
    """Extract a ``Retry-After`` delay (seconds) from an httpx status error if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    return _parse_retry_after(raw)


async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args: object,
    max_retries: int = 0,
    delay: float = 1.0,
    **kwargs: object,
) -> T:
    """
    Call *fn* with exponential backoff and jitter.

    Only retries transient failures — timeouts, connection errors, and HTTP
    429/5xx responses. Non-retryable errors (4xx such as auth/bad-request, and
    any error that is not clearly transient) are re-raised immediately.

    If a retryable ``httpx.HTTPStatusError`` carries a ``Retry-After`` header,
    that delay is respected instead of the computed backoff.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            if attempt < max_retries:
                retry_after = _retry_after_from_exc(exc)
                if retry_after is not None:
                    sleep_for = retry_after
                else:
                    base = delay * (2**attempt)
                    # Full jitter: sleep a random amount in [0, base].
                    sleep_for = random.uniform(0, base)
                await asyncio.sleep(sleep_for)
    raise last_exc  # type: ignore[misc]
