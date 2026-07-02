from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def requires_human_confirmation(
    reason: str | F | None = None,
) -> Any:
    """Mark an action helper as requiring explicit human confirmation.

    Can be used as ``@requires_human_confirmation`` (no args) or
    ``@requires_human_confirmation("reason")`` (with a string reason).
    """

    def _wrap(func: F, _reason: str | None) -> F:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)

            async_wrapper.__synapsekit_requires_confirmation__ = True  # type: ignore[attr-defined]
            async_wrapper.__synapsekit_confirmation_reason__ = _reason  # type: ignore[attr-defined]
            async_wrapper._requires_confirmation = True  # type: ignore[attr-defined]
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            sync_wrapper.__synapsekit_requires_confirmation__ = True  # type: ignore[attr-defined]
            sync_wrapper.__synapsekit_confirmation_reason__ = _reason  # type: ignore[attr-defined]
            sync_wrapper._requires_confirmation = True  # type: ignore[attr-defined]
            return cast(F, sync_wrapper)

    # Called as @requires_human_confirmation (no parentheses) — reason is the function
    if callable(reason):
        return _wrap(cast(F, reason), None)

    # Called as @requires_human_confirmation() or @requires_human_confirmation("text")
    def decorator(func: F) -> F:
        return _wrap(func, cast("str | None", reason))

    return decorator


def requires_confirmation(obj: Any) -> bool:
    """Return whether an object was marked with ``requires_human_confirmation``."""

    return bool(getattr(obj, "__synapsekit_requires_confirmation__", False))
